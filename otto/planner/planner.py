"""
Otto planner/planner.py
-----------------------
Translates a WorkflowTemplate into concrete DB records for one run.

The planner is the bridge between the model layer (pure Pydantic) and the store
layer (SQLModel / SQLite). It creates:

  - One Run record (status PENDING)
  - One TaskRun per task template:
      - Root tasks (no dependencies) → status READY
      - All others                   → status PENDING
      - Inputs serialized as wiring specs so the executor can resolve values
        without the original template
      - Outputs serialized as output declarations so the executor knows what to
        produce and register as artifacts
  - One static Edge per declared dependency (kind STATIC, status PENDING)
  - RUN_CREATED and TASK_CREATED events

No dynamic fan-out (shard creation) happens here. The scheduler's advance()
creates shard TaskRuns at runtime when an upstream task produces array artifacts.

The caller is responsible for session.commit() / rollback().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlmodel import Session

from otto.model.workflow import WorkflowTemplate
from otto.store.queries import (
    create_edge,
    create_run,
    create_task_run,
    emit_event,
)
from otto.store.tables import (
    Edge,
    EventEntityType,
    EventType,
    Run,
    TaskRun,
    TaskRunStatus,
)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class PlanResult:
    """All DB records created by a single plan_run() call."""

    run:       Run
    task_runs: dict[str, TaskRun]  # task name → TaskRun
    edges:     list[Edge] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def _validate_parameters(
    template: WorkflowTemplate,
    parameters: dict[str, Any],
) -> None:
    """
    Raise ValueError if the supplied parameters are incompatible with the
    workflow's declared inputs.

    Checks:
      - No parameter names that are not declared in the workflow
      - Every required input has a supplied value (default counts as supplied
        only if the caller omits the key entirely; the planner does not
        auto-apply defaults — that is the CLI's responsibility)
    """
    unknown = set(parameters) - set(template.inputs)
    if unknown:
        raise ValueError(
            f"Unknown parameters for workflow '{template.name}': {sorted(unknown)}"
        )

    missing = [
        name
        for name, decl in template.inputs.items()
        if decl.required and name not in parameters
    ]
    if missing:
        raise ValueError(
            f"Required inputs not supplied for workflow '{template.name}': "
            f"{sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_run(
    session: Session,
    template: WorkflowTemplate,
    *,
    parameters: dict[str, Any] | None = None,
    label: str | None = None,
) -> PlanResult:
    """
    Translate a WorkflowTemplate into a concrete Run in the database.

    Args:
        session:    SQLModel session. Caller commits or rolls back.
        template:   Validated WorkflowTemplate (model layer).
        parameters: User-supplied values for workflow-level inputs.
                    Required inputs must be present. Unknown keys are rejected.
        label:      Optional human-readable tag stored on the Run record.

    Returns:
        PlanResult with the created Run, a name→TaskRun mapping, and edge list.

    Raises:
        ValueError: if required parameters are missing or unknown keys are supplied.
    """
    params = parameters or {}
    _validate_parameters(template, params)

    # ------------------------------------------------------------------ Run --
    run = create_run(
        session,
        workflow_name=template.name,
        workflow_version=template.version,
        parameters=params,
        label=label,
    )
    session.flush()  # populate run.id before creating TaskRuns

    # --------------------------------------------------------------- TaskRuns --
    # Pass 1: create all TaskRun rows. We need them all flushed before we can
    # reference their IDs when building edges.
    task_runs: dict[str, TaskRun] = {}

    for task in template.tasks:
        # Root tasks have no ordering constraints — they can be submitted immediately.
        status = (
            TaskRunStatus.READY
            if not task.dependencies
            else TaskRunStatus.PENDING
        )

        # Serialize the wiring spec for each input slot. The executor reads this
        # at submission time to resolve actual values from run parameters and
        # upstream artifact URIs.
        inputs_snapshot: dict[str, Any] = {
            slot: spec.model_dump()
            for slot, spec in task.inputs.items()
        }

        # Serialize output declarations so the executor knows what artifacts to
        # register after completion, without needing the original template.
        outputs_snapshot: dict[str, Any] = {
            slot: spec.model_dump()
            for slot, spec in task.outputs.items()
        }

        # Serialize resource profiles as plain dicts (JSON column).
        resource_profiles = [p.model_dump() for p in task.resources.profiles]

        tr = create_task_run(
            session,
            run_id=run.id,
            name=task.name,
            engine=task.engine,
            backend=task.backend,
            cmd=task.cmd,
            resource_profiles=resource_profiles or None,
            escalate_on=task.resources.escalate_on or None,
            inputs=inputs_snapshot or None,
            outputs=outputs_snapshot or None,
            scatter=task.scatter,
            # scatter_method is only meaningful when scatter is set
            scatter_method=task.scatter_method if task.scatter else None,
            status=status,
        )
        task_runs[task.name] = tr

    session.flush()  # populate all task_run IDs before building edges

    # ----------------------------------------------------------------- Edges --
    # Pass 2: one STATIC edge per declared dependency. Edge represents an ordering
    # constraint only; the wiring detail (which output feeds which input) lives in
    # TaskRun.inputs and does not need to be duplicated on the edge.
    edges: list[Edge] = []

    for task in template.tasks:
        downstream_tr = task_runs[task.name]
        for dep_name in task.dependencies:
            upstream_tr = task_runs[dep_name]
            edge = create_edge(
                session,
                run_id=run.id,
                upstream_task_run_id=upstream_tr.id,
                downstream_task_run_id=downstream_tr.id,
            )
            edges.append(edge)

    # --------------------------------------------------------------- Events --
    emit_event(
        session,
        entity_type=EventEntityType.RUN,
        entity_id=run.id,
        run_id=run.id,
        event_type=EventType.RUN_CREATED,
    )
    for tr in task_runs.values():
        emit_event(
            session,
            entity_type=EventEntityType.TASK_RUN,
            entity_id=tr.id,
            run_id=run.id,
            event_type=EventType.TASK_CREATED,
        )

    return PlanResult(run=run, task_runs=task_runs, edges=edges)
