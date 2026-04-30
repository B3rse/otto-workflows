"""
Otto store/tables.py
--------------------
SQLModel table definitions for all Otto runtime objects.

Design notes:
- Shards are TaskRuns with parent_task_run_id set (no separate ShardRun table)
- Edges support both STATIC (plan-time) and DYNAMIC (runtime fan-out) kinds
- Artifacts track array membership so downstream tasks can gather shard outputs
- Resource profiles are stored as JSON arrays on TaskRun; escalation is driven
  by incrementing resource_profile_index on retry
- Events table is append-only; never update or delete rows from it
- Leases enable safe concurrent advance() calls (foreground loop + pull model)
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Text
from sqlmodel import Column, Field, Index, SQLModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_id() -> str:
    return str(uuid.uuid4())

def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RunStatus(str, enum.Enum):
    PENDING   = "PENDING"
    RUNNING   = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED    = "FAILED"
    CANCELLED = "CANCELLED"


class TaskRunStatus(str, enum.Enum):
    PENDING        = "PENDING"
    PENDING_SHARDS = "PENDING_SHARDS"  # waiting for upstream array to materialise
    READY          = "READY"
    SUBMITTING     = "SUBMITTING"
    SUBMITTED      = "SUBMITTED"
    RUNNING        = "RUNNING"
    SUCCEEDED      = "SUCCEEDED"
    FAILED         = "FAILED"
    WAITING_RETRY  = "WAITING_RETRY"   # resource escalation scheduled
    SKIPPED        = "SKIPPED"
    CANCELLED      = "CANCELLED"


class EdgeKind(str, enum.Enum):
    STATIC  = "STATIC"   # created by planner
    DYNAMIC = "DYNAMIC"  # created at runtime during fan-out


class EdgeStatus(str, enum.Enum):
    PENDING  = "PENDING"   # upstream not yet complete
    RESOLVED = "RESOLVED"  # upstream succeeded; downstream may proceed


class ArtifactKind(str, enum.Enum):
    FILE      = "FILE"
    DIRECTORY = "DIRECTORY"
    VALUE     = "VALUE"   # scalar in-memory / DB-stored value
    ARRAY     = "ARRAY"   # virtual: groups array elements (no URI itself)


class ExternalJobStatus(str, enum.Enum):
    SUBMITTED = "SUBMITTED"
    RUNNING   = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED    = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN   = "UNKNOWN"


class EventEntityType(str, enum.Enum):
    RUN      = "RUN"
    TASK_RUN = "TASK_RUN"
    ARTIFACT = "ARTIFACT"
    EDGE     = "EDGE"
    LEASE    = "LEASE"


class EventType(str, enum.Enum):
    # Run lifecycle
    RUN_CREATED   = "RUN_CREATED"
    RUN_STARTED   = "RUN_STARTED"
    RUN_SUCCEEDED = "RUN_SUCCEEDED"
    RUN_FAILED    = "RUN_FAILED"
    RUN_CANCELLED = "RUN_CANCELLED"
    # TaskRun lifecycle
    TASK_CREATED        = "TASK_CREATED"
    TASK_READY          = "TASK_READY"
    TASK_SUBMITTED      = "TASK_SUBMITTED"
    TASK_RUNNING        = "TASK_RUNNING"
    TASK_SUCCEEDED      = "TASK_SUCCEEDED"
    TASK_FAILED         = "TASK_FAILED"
    TASK_RETRY_QUEUED   = "TASK_RETRY_QUEUED"
    TASK_ESCALATED      = "TASK_ESCALATED"    # resource profile bumped
    TASK_SHARDS_SPAWNED = "TASK_SHARDS_SPAWNED"
    TASK_SKIPPED        = "TASK_SKIPPED"
    TASK_CANCELLED      = "TASK_CANCELLED"
    # Artifact
    ARTIFACT_CREATED = "ARTIFACT_CREATED"
    # Edge
    EDGE_RESOLVED = "EDGE_RESOLVED"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

class Run(SQLModel, table=True):
    """
    Top-level record for one execution of a workflow template.

    Fields:
        workflow_name:    template that was used (e.g. "rna_seq_pipeline")
        workflow_version: optional version of that template
        status:           PENDING → RUNNING → SUCCEEDED / FAILED / CANCELLED
        parameters:       user-supplied inputs (e.g. {"sample": "S1"})
        label:            optional human-readable tag for this run
        started_at:       when the first task started
        finished_at:      when the run completed (success or failure)
    """

    __tablename__ = "runs"

    id:               str            = Field(default_factory=_new_id, primary_key=True)
    workflow_name:    str            = Field(index=True)
    workflow_version: str | None    = Field(default=None)
    status:           RunStatus     = Field(default=RunStatus.PENDING, index=True)
    parameters:       dict | None   = Field(default=None, sa_column=Column(JSON))
    label:            str | None    = Field(default=None)
    created_at:       datetime      = Field(default_factory=_now)
    started_at:       datetime | None = Field(default=None)
    finished_at:      datetime | None = Field(default=None)


class TaskRun(SQLModel, table=True):
    """
    A single executable unit within a Run.

    Shards are TaskRuns with parent_task_run_id set; shard_index / shard_key
    are null for non-shard tasks.

    Fields:
        name:                   template task name — not unique across runs
        parent_task_run_id:     the task that spawned this shard (shards only)
        shard_index / key:      position and optional label within the array
        engine:                 what runs the cmd: shell | nextflow | cwl | ...
        backend:                where it runs: local | slurm | aws_batch | ...
        resource_profiles:      ordered list of [{cpus, mem_gb, walltime}, ...]
        resource_profile_index: active profile (0-based); incremented on escalation
        escalate_on:            failure reasons that trigger escalation e.g. ["OOM"]
        attempt:                how many times this task has been attempted
        inputs / outputs:       data-flow snapshots, self-contained per task
        backend_config:         full backend config snapshot at submission time
        ready_at:               when all dependencies resolved
        submitted_at:           when handed to the backend
        started_at:             when the backend reported it actually started
        finished_at:            when it completed (success or failure)
    """

    __tablename__ = "task_runs"
    __table_args__ = (Index("ix_task_runs_run_status", "run_id", "status"),)

    id:     str = Field(default_factory=_new_id, primary_key=True)
    run_id: str = Field(index=True, foreign_key="runs.id")
    name:   str = Field(index=True)

    # Sharding
    parent_task_run_id: str | None = Field(
        default=None, index=True, foreign_key="task_runs.id"
    )
    shard_index: int | None = Field(default=None)
    shard_key:   str | None = Field(default=None)

    # Execution
    engine: str      = Field(default="shell")
    backend: str     = Field(default="local")
    cmd: str | None  = Field(default=None, sa_column=Column(Text))

    # Resource escalation
    resource_profiles:      list | None = Field(default=None, sa_column=Column(JSON))
    resource_profile_index: int         = Field(default=0)
    escalate_on:            list | None = Field(default=None, sa_column=Column(JSON))

    # Runtime state
    status:  TaskRunStatus = Field(default=TaskRunStatus.PENDING, index=True)
    attempt: int           = Field(default=0)

    # Data flow
    inputs:         dict | None = Field(default=None, sa_column=Column(JSON))
    outputs:        dict | None = Field(default=None, sa_column=Column(JSON))
    backend_config: dict | None = Field(default=None, sa_column=Column(JSON))

    # Timestamps
    created_at:   datetime       = Field(default_factory=_now)
    ready_at:     datetime | None = Field(default=None)
    submitted_at: datetime | None = Field(default=None)
    started_at:   datetime | None = Field(default=None)
    finished_at:  datetime | None = Field(default=None)


class Edge(SQLModel, table=True):
    """
    Directed dependency edge between two TaskRuns.

    STATIC edges are created by the planner before execution starts.
    DYNAMIC edges are created at runtime when an upstream task emits an array
    artifact and the scheduler fans out new shard TaskRuns.

    Fields:
        upstream_task_run_id:   the task that must finish first
        downstream_task_run_id: the task that is waiting
        kind:                   STATIC (planner) or DYNAMIC (runtime fan-out)
        status:                 PENDING → RESOLVED when upstream succeeds
        output_name:            which output slot of upstream feeds this edge
        input_name:             which input slot of downstream receives it
        resolved_at:            when upstream finished and this edge was resolved
    """

    __tablename__ = "edges"
    __table_args__ = (
        Index("ix_edges_downstream", "downstream_task_run_id", "status"),
        Index("ix_edges_upstream", "upstream_task_run_id"),
    )

    id:                     str            = Field(default_factory=_new_id, primary_key=True)
    run_id:                 str            = Field(index=True, foreign_key="runs.id")
    upstream_task_run_id:   str            = Field(foreign_key="task_runs.id")
    downstream_task_run_id: str            = Field(foreign_key="task_runs.id")
    kind:                   EdgeKind       = Field(default=EdgeKind.STATIC)
    status:                 EdgeStatus     = Field(default=EdgeStatus.PENDING, index=True)
    output_name:            str | None     = Field(default=None)
    input_name:             str | None     = Field(default=None)
    created_at:             datetime       = Field(default_factory=_now)
    resolved_at:            datetime | None = Field(default=None)


class Artifact(SQLModel, table=True):
    """
    Any output produced by a TaskRun.

    Array outputs are represented as:
      - One ARRAY-kind parent artifact  (is_array=True, array_index=None, uri=None)
      - N child artifacts               (is_array=True, array_index=i, array_parent_id=parent.id)

    Downstream tasks reference the parent ID; the scheduler queries children to
    fan out one shard per element.

    Fields:
        produced_by_task_run_id: which TaskRun produced this
        name:                    logical name (e.g. "aligned_bam")
        kind:                    FILE | DIRECTORY | VALUE | ARRAY
        uri:                     file path or location; null for ARRAY parents
        value:                   used instead of uri for scalar VALUE artifacts
        is_array:                True for both the parent row and all child rows
        array_index:             position within the array; null on the parent row
        array_parent_id:         points to the parent ARRAY row; null on the parent

    Example — 3 BAM files produce 4 rows:
    ┌────┬───────────┬───────┬─────────┬──────────┬─────────────┬─────────────────┐
    │ id │   name    │ kind  │   uri   │ is_array │ array_index │ array_parent_id │
    ├────┼───────────┼───────┼─────────┼──────────┼─────────────┼─────────────────┤
    │ p1 │ bam_files │ ARRAY │ null    │ true     │ null        │ null            │
    │ c1 │ bam_files │ FILE  │ /s1.bam │ true     │ 0           │ p1              │
    │ c2 │ bam_files │ FILE  │ /s2.bam │ true     │ 1           │ p1              │
    │ c3 │ bam_files │ FILE  │ /s3.bam │ true     │ 2           │ p1              │
    └────┴───────────┴───────┴─────────┴──────────┴─────────────┴─────────────────┘
    """

    __tablename__ = "artifacts"
    __table_args__ = (Index("ix_artifacts_parent", "array_parent_id", "array_index"),)

    id:                      str            = Field(default_factory=_new_id, primary_key=True)
    run_id:                  str            = Field(index=True, foreign_key="runs.id")
    produced_by_task_run_id: str            = Field(index=True, foreign_key="task_runs.id")
    name:                    str            = Field(index=True)
    kind:                    ArtifactKind   = Field(default=ArtifactKind.FILE)
    uri:                     str | None     = Field(default=None)
    value:                   str | None     = Field(default=None)
    is_array:                bool           = Field(default=False)
    array_index:             int | None     = Field(default=None)
    array_parent_id:         str | None     = Field(default=None, foreign_key="artifacts.id")
    created_at:              datetime       = Field(default_factory=_now)


class ExternalJob(SQLModel, table=True):
    """
    Tracks one job submission to an external backend.
    A TaskRun can have multiple ExternalJob rows across retry attempts.

    Fields:
        run_id:           denormalised for fast run-level polling queries
        attempt:          matches TaskRun.attempt for this submission
        ext_id:           backend's own ID: Slurm job ID, PID, AWS ARN, ...
        status:           SUBMITTED → RUNNING → SUCCEEDED / FAILED / CANCELLED
        exit_code:        process exit code (0 = success)
        exit_reason:      OOM | TIMEOUT | USER — checked against TaskRun.escalate_on
        logs_uri:         where to find logs for this job
        resource_profile: snapshot of the resource profile used for this attempt
    """

    __tablename__ = "external_jobs"

    id:               str                = Field(default_factory=_new_id, primary_key=True)
    task_run_id:      str                = Field(index=True, foreign_key="task_runs.id")
    run_id:           str                = Field(index=True, foreign_key="runs.id")
    backend:          str                = Field()
    attempt:          int                = Field(default=0)
    ext_id:           str                = Field(index=True)
    status:           ExternalJobStatus  = Field(default=ExternalJobStatus.SUBMITTED)
    exit_code:        int | None         = Field(default=None)
    exit_reason:      str | None         = Field(default=None)
    logs_uri:         str | None         = Field(default=None)
    submitted_at:     datetime           = Field(default_factory=_now)
    started_at:       datetime | None    = Field(default=None)
    finished_at:      datetime | None    = Field(default=None)
    resource_profile: dict | None        = Field(default=None, sa_column=Column(JSON))


class Event(SQLModel, table=True):
    """
    Append-only audit log. Never update or delete rows.
    Every state change writes a row here, giving a full timeline of each run.

    Fields:
        entity_type: what kind of thing this event is about: RUN | TASK_RUN | ...
        entity_id:   ID of that specific thing
        run_id:      which Run this event belongs to (for fast filtering)
        event_type:  what happened: TASK_SUCCEEDED | TASK_FAILED | ...
        payload:     event-specific context (e.g. {"exit_code": 1, "exit_reason": "OOM"})
    """

    __tablename__ = "events"

    id:          str              = Field(default_factory=_new_id, primary_key=True)
    timestamp:   datetime         = Field(default_factory=_now, index=True)
    entity_type: EventEntityType  = Field(index=True)
    entity_id:   str              = Field(index=True)
    run_id:      str | None       = Field(default=None, index=True)
    event_type:  EventType        = Field(index=True)
    payload:     dict | None      = Field(default=None, sa_column=Column(JSON))


class Lease(SQLModel, table=True):
    """
    Optimistic lock for advance() and job submission.
    Prevents two concurrent callers from double-submitting the same TaskRun.
    Leases expire automatically so a crashed process can't hold a lock forever.

    Fields:
        entity_type: what kind of thing is locked: "task_run" | "run"
        entity_id:   ID of the locked thing
        owner:       which process holds the lock (e.g. "loop:pid:12345")
        acquired_at: when the lease was taken
        expires_at:  TTL — after this, another process may take over
    """

    __tablename__ = "leases"

    id:          str      = Field(default_factory=_new_id, primary_key=True)
    entity_type: str      = Field(index=True)
    entity_id:   str      = Field(index=True)
    owner:       str      = Field()
    acquired_at: datetime = Field(default_factory=_now)
    expires_at:  datetime = Field()
