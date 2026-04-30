"""
Otto store package.
Public API — import everything you need from here.
"""

from otto.store.db import get_engine, get_session, init_db
from otto.store.queries import (
    # Leases
    acquire_lease,
    all_upstream_edges_resolved,
    cleanup_expired_leases,
    create_array_artifact,
    # Artifact
    create_artifact,
    # Edge
    create_edge,
    # ExternalJob
    create_external_job,
    # Run
    create_run,
    # TaskRun
    create_task_run,
    # Events
    emit_event,
    get_active_external_jobs,
    get_active_resource_profile,
    get_array_elements,
    get_artifact_by_name,
    get_artifacts_for_task_run,
    get_edges_for_downstream,
    get_edges_for_upstream,
    get_events,
    get_external_jobs_for_task_run,
    get_ready_task_runs,
    get_run,
    get_runs,
    get_shards_for_task,
    get_submitted_task_runs,
    get_task_run,
    get_task_runs_for_run,
    increment_task_run_attempt,
    release_lease,
    resolve_edge,
    update_external_job,
    update_run_status,
    update_task_run_status,
)
from otto.store.tables import (
    Artifact,
    ArtifactKind,
    Edge,
    EdgeKind,
    EdgeStatus,
    Event,
    EventEntityType,
    EventType,
    ExternalJob,
    ExternalJobStatus,
    Lease,
    Run,
    RunStatus,
    TaskRun,
    TaskRunStatus,
)

__all__ = [
    # db
    "get_engine", "get_session", "init_db",
    # tables
    "Artifact", "ArtifactKind",
    "Edge", "EdgeKind", "EdgeStatus",
    "Event", "EventEntityType", "EventType",
    "ExternalJob", "ExternalJobStatus",
    "Lease",
    "Run", "RunStatus",
    "TaskRun", "TaskRunStatus",
    # queries
    "create_run", "get_run", "get_runs", "update_run_status",
    "create_task_run", "get_task_run", "get_task_runs_for_run",
    "get_shards_for_task", "get_ready_task_runs", "get_submitted_task_runs",
    "update_task_run_status", "increment_task_run_attempt", "get_active_resource_profile",
    "create_edge", "get_edges_for_downstream", "get_edges_for_upstream",
    "resolve_edge", "all_upstream_edges_resolved",
    "create_artifact", "create_array_artifact", "get_artifacts_for_task_run",
    "get_array_elements", "get_artifact_by_name",
    "create_external_job", "get_external_jobs_for_task_run",
    "get_active_external_jobs", "update_external_job",
    "emit_event", "get_events",
    "acquire_lease", "release_lease", "cleanup_expired_leases",
]
