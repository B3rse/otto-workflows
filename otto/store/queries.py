"""
Otto store/queries.py
---------------------
All database read/write operations.

Rules:
- Every function accepts a Session as its first argument.
- No business logic here — just DB operations.
- Callers are responsible for commit() / rollback().
- Functions that return lists always return [] rather than None.
- Timestamps are always UTC.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlmodel import Session, col, select

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

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def create_run(
    session: Session,
    *,
    workflow_name: str,
    workflow_version: str | None = None,
    parameters: dict | None = None,
    label: str | None = None,
) -> Run:
    run = Run(
        workflow_name=workflow_name,
        workflow_version=workflow_version,
        parameters=parameters or {},
        label=label,
    )
    session.add(run)
    return run


def get_run(session: Session, run_id: str) -> Run | None:
    return session.get(Run, run_id)


def get_runs(
    session: Session,
    *,
    status: RunStatus | None = None,
    workflow_name: str | None = None,
    limit: int = 100, # TODO: will need to add offset for pagination to support full table scanning
) -> list[Run]:
    stmt = select(Run)
    if status is not None:
        stmt = stmt.where(Run.status == status)
    if workflow_name is not None:
        stmt = stmt.where(Run.workflow_name == workflow_name)
    stmt = stmt.order_by(col(Run.created_at).desc()).limit(limit)
    return list(session.exec(stmt).all())


def update_run_status(
    session: Session,
    run_id: str,
    status: RunStatus,
    *,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> Run:
    run = session.get(Run, run_id)
    if run is None:
        raise ValueError(f"Run not found: {run_id}")
    run.status = status
    if started_at is not None:
        run.started_at = started_at
    if finished_at is not None:
        run.finished_at = finished_at
    session.add(run)
    return run


# ---------------------------------------------------------------------------
# TaskRun
# ---------------------------------------------------------------------------

def create_task_run(
    session: Session,
    *,
    run_id: str,
    name: str,
    engine: str = "shell",
    backend: str = "local",
    cmd: str | None = None,
    resource_profiles: list | None = None,
    escalate_on: list | None = None,
    inputs: dict | None = None,
    outputs: dict | None = None,
    scatter: str | None = None,
    scatter_method: str | None = None,
    parent_task_run_id: str | None = None,
    shard_index: int | None = None,
    shard_key: str | None = None,
    status: TaskRunStatus = TaskRunStatus.PENDING,
) -> TaskRun:
    tr = TaskRun(
        run_id=run_id,
        name=name,
        engine=engine,
        backend=backend,
        cmd=cmd,
        resource_profiles=resource_profiles or [],  # None → [] (no profiles)
        escalate_on=escalate_on,
        inputs=inputs or {},   # None → {} (no wiring — executor sees empty dict)
        outputs=outputs,       # None kept as-is — None means "no outputs declared"
        scatter=scatter,
        scatter_method=scatter_method,
        parent_task_run_id=parent_task_run_id,
        shard_index=shard_index,
        shard_key=shard_key,
        status=status,
    )
    session.add(tr)
    return tr


def get_task_run(session: Session, task_run_id: str) -> TaskRun | None:
    return session.get(TaskRun, task_run_id)


def get_task_runs_for_run(
    session: Session,
    run_id: str,
    *,
    status: TaskRunStatus | None = None,
) -> list[TaskRun]:
    stmt = select(TaskRun).where(TaskRun.run_id == run_id)
    if status is not None:
        stmt = stmt.where(TaskRun.status == status)
    return list(session.exec(stmt).all())


def get_shards_for_task(session: Session, parent_task_run_id: str) -> list[TaskRun]:
    stmt = select(TaskRun).where(TaskRun.parent_task_run_id == parent_task_run_id)
    return list(session.exec(stmt).all())


def get_ready_task_runs(session: Session, run_id: str) -> list[TaskRun]:
    stmt = (
        select(TaskRun)
        .where(TaskRun.run_id == run_id)
        .where(TaskRun.status == TaskRunStatus.READY)
    )
    return list(session.exec(stmt).all())


def get_submitted_task_runs(session: Session, run_id: str) -> list[TaskRun]:
    stmt = (
        select(TaskRun)
        .where(TaskRun.run_id == run_id)
        .where(TaskRun.status.in_([  # type: ignore[attr-defined]
            TaskRunStatus.SUBMITTED,
            TaskRunStatus.SUBMITTING,
            TaskRunStatus.RUNNING,
        ]))
    )
    return list(session.exec(stmt).all())


def update_task_run_status(
    session: Session,
    task_run_id: str,
    status: TaskRunStatus,
    *,
    ready_at: datetime | None = None,
    submitted_at: datetime | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    outputs: dict | None = None,
    backend_config: dict | None = None,
) -> TaskRun:
    tr = session.get(TaskRun, task_run_id)
    if tr is None:
        raise ValueError(f"TaskRun not found: {task_run_id}")
    tr.status = status
    if ready_at is not None:
        tr.ready_at = ready_at
    if submitted_at is not None:
        tr.submitted_at = submitted_at
    if started_at is not None:
        tr.started_at = started_at
    if finished_at is not None:
        tr.finished_at = finished_at
    if outputs is not None:
        tr.outputs = outputs
    if backend_config is not None:
        tr.backend_config = backend_config
    session.add(tr)
    return tr


def increment_task_run_attempt(
    session: Session,
    task_run_id: str,
    *,
    escalate_profile: bool = False,
) -> TaskRun:
    """
    Bump attempt counter. Optionally advance to the next resource profile.
    Raises ValueError if escalation is requested but no next profile exists.
    """
    tr = session.get(TaskRun, task_run_id)
    if tr is None:
        raise ValueError(f"TaskRun not found: {task_run_id}")
    tr.attempt += 1
    if escalate_profile:
        profiles = tr.resource_profiles or []
        next_index = tr.resource_profile_index + 1
        if next_index >= len(profiles):
            raise ValueError(
                f"No next resource profile for task_run {task_run_id} "
                f"(current index {tr.resource_profile_index}, "
                f"{len(profiles)} profiles total)"
            )
        tr.resource_profile_index = next_index
    session.add(tr)
    return tr


def get_active_resource_profile(task_run: TaskRun) -> dict | None:
    """Return the currently active resource profile dict, or None."""
    profiles = task_run.resource_profiles or []
    if not profiles:
        return None
    idx = task_run.resource_profile_index
    return profiles[idx] if idx < len(profiles) else profiles[-1]


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

def create_edge(
    session: Session,
    *,
    run_id: str,
    upstream_task_run_id: str,
    downstream_task_run_id: str,
    kind: EdgeKind = EdgeKind.STATIC,
    output_name: str | None = None,
    input_name: str | None = None,
) -> Edge:
    edge = Edge(
        run_id=run_id,
        upstream_task_run_id=upstream_task_run_id,
        downstream_task_run_id=downstream_task_run_id,
        kind=kind,
        output_name=output_name,
        input_name=input_name,
    )
    session.add(edge)
    return edge


def get_edges_for_downstream(session: Session, downstream_task_run_id: str) -> list[Edge]:
    stmt = select(Edge).where(Edge.downstream_task_run_id == downstream_task_run_id)
    return list(session.exec(stmt).all())


def get_edges_for_upstream(session: Session, upstream_task_run_id: str) -> list[Edge]:
    stmt = select(Edge).where(Edge.upstream_task_run_id == upstream_task_run_id)
    return list(session.exec(stmt).all())


def resolve_edge(session: Session, edge_id: str) -> Edge:
    edge = session.get(Edge, edge_id)
    if edge is None:
        raise ValueError(f"Edge not found: {edge_id}")
    edge.status = EdgeStatus.RESOLVED
    edge.resolved_at = datetime.now(UTC).replace(tzinfo=None)
    session.add(edge)
    return edge


def all_upstream_edges_resolved(session: Session, task_run_id: str) -> bool:
    """Return True iff every incoming edge for this TaskRun is RESOLVED."""
    edges = get_edges_for_downstream(session, task_run_id)
    if not edges:
        return True  # no dependencies → immediately runnable
    return all(e.status == EdgeStatus.RESOLVED for e in edges)


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------

def create_artifact(
    session: Session,
    *,
    run_id: str,
    produced_by_task_run_id: str,
    name: str,
    kind: ArtifactKind = ArtifactKind.FILE,
    uri: str | None = None,
    value: str | None = None,
    is_array: bool = False,
    array_index: int | None = None,
    array_parent_id: str | None = None,
) -> Artifact:
    artifact = Artifact(
        run_id=run_id,
        produced_by_task_run_id=produced_by_task_run_id,
        name=name,
        kind=kind,
        uri=uri,
        value=value,
        is_array=is_array,
        array_index=array_index,
        array_parent_id=array_parent_id,
    )
    session.add(artifact)
    return artifact


def create_array_artifact(
    session: Session,
    *,
    run_id: str,
    produced_by_task_run_id: str,
    name: str,
    elements: list[str],   # list of URIs or values
    kind: ArtifactKind = ArtifactKind.FILE,
) -> tuple[Artifact, list[Artifact]]:
    """
    Create an ARRAY parent artifact and N child artifacts in one call.
    Returns (parent, [child, ...]).
    """
    parent = create_artifact(
        session,
        run_id=run_id,
        produced_by_task_run_id=produced_by_task_run_id,
        name=name,
        kind=ArtifactKind.ARRAY,
        is_array=True,
    )
    session.flush()  # populate parent.id

    children: list[Artifact] = []
    for i, element in enumerate(elements):
        child = create_artifact(
            session,
            run_id=run_id,
            produced_by_task_run_id=produced_by_task_run_id,
            name=name,
            kind=kind,
            uri=element if kind != ArtifactKind.VALUE else None,
            value=element if kind == ArtifactKind.VALUE else None,
            is_array=True,
            array_index=i,
            array_parent_id=parent.id,
        )
        children.append(child)

    return parent, children


def get_artifacts_for_task_run(session: Session, task_run_id: str) -> list[Artifact]:
    stmt = select(Artifact).where(Artifact.produced_by_task_run_id == task_run_id)
    return list(session.exec(stmt).all())


def get_array_elements(session: Session, array_parent_id: str) -> list[Artifact]:
    stmt = (
        select(Artifact)
        .where(Artifact.array_parent_id == array_parent_id)
        .order_by(col(Artifact.array_index))
    )
    return list(session.exec(stmt).all())


def get_artifact_by_name(
    session: Session,
    task_run_id: str,
    name: str,
) -> Artifact | None:
    stmt = (
        select(Artifact)
        .where(Artifact.produced_by_task_run_id == task_run_id)
        .where(Artifact.name == name)
        .where(Artifact.array_index == None)  # noqa: E711 — SQLAlchemy idiom
    )
    return session.exec(stmt).first()


# ---------------------------------------------------------------------------
# ExternalJob
# ---------------------------------------------------------------------------

def create_external_job(
    session: Session,
    *,
    task_run_id: str,
    run_id: str,
    backend: str,
    ext_id: str,
    attempt: int = 0,
    resource_profile: dict | None = None,
    logs_uri: str | None = None,
) -> ExternalJob:
    job = ExternalJob(
        task_run_id=task_run_id,
        run_id=run_id,
        backend=backend,
        ext_id=ext_id,
        attempt=attempt,
        resource_profile=resource_profile,
        logs_uri=logs_uri,
    )
    session.add(job)
    return job


def get_external_jobs_for_task_run(
    session: Session,
    task_run_id: str,
) -> list[ExternalJob]:
    stmt = (
        select(ExternalJob)
        .where(ExternalJob.task_run_id == task_run_id)
        .order_by(col(ExternalJob.submitted_at))
    )
    return list(session.exec(stmt).all())


def get_active_external_jobs(session: Session, run_id: str) -> list[ExternalJob]:
    """All submitted/running jobs for a run (used for batch polling)."""
    stmt = (
        select(ExternalJob)
        .where(ExternalJob.run_id == run_id)
        .where(ExternalJob.status.in_([  # type: ignore[attr-defined]
            ExternalJobStatus.SUBMITTED,
            ExternalJobStatus.RUNNING,
        ]))
    )
    return list(session.exec(stmt).all())


def update_external_job(
    session: Session,
    external_job_id: str,
    *,
    status: ExternalJobStatus | None = None,
    exit_code: int | None = None,
    exit_reason: str | None = None,
    logs_uri: str | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> ExternalJob:
    job = session.get(ExternalJob, external_job_id)
    if job is None:
        raise ValueError(f"ExternalJob not found: {external_job_id}")
    if status is not None:
        job.status = status
    if exit_code is not None:
        job.exit_code = exit_code
    if exit_reason is not None:
        job.exit_reason = exit_reason
    if logs_uri is not None:
        job.logs_uri = logs_uri
    if started_at is not None:
        job.started_at = started_at
    if finished_at is not None:
        job.finished_at = finished_at
    session.add(job)
    return job


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

def emit_event(
    session: Session,
    *,
    entity_type: EventEntityType,
    entity_id: str,
    event_type: EventType,
    run_id: str | None = None,
    payload: dict | None = None,
) -> Event:
    event = Event(
        entity_type=entity_type,
        entity_id=entity_id,
        run_id=run_id,
        event_type=event_type,
        payload=payload or {},
    )
    session.add(event)
    return event


def get_events(
    session: Session,
    *,
    run_id: str | None = None,
    entity_id: str | None = None,
    entity_type: EventEntityType | None = None,
    event_type: EventType | None = None,
    limit: int = 200,
) -> list[Event]:
    stmt = select(Event)
    if run_id is not None:
        stmt = stmt.where(Event.run_id == run_id)
    if entity_id is not None:
        stmt = stmt.where(Event.entity_id == entity_id)
    if entity_type is not None:
        stmt = stmt.where(Event.entity_type == entity_type)
    if event_type is not None:
        stmt = stmt.where(Event.event_type == event_type)
    stmt = stmt.order_by(col(Event.timestamp).desc()).limit(limit)
    return list(session.exec(stmt).all())


# ---------------------------------------------------------------------------
# Leases
# ---------------------------------------------------------------------------

def acquire_lease(
    session: Session,
    *,
    entity_type: str,
    entity_id: str,
    owner: str,
    ttl_seconds: int = 300,
) -> Lease | None:
    """
    Try to acquire a lease. Returns the Lease on success, None if already held
    by another owner with a non-expired lease.
    Expired leases are silently replaced.
    """
    now = datetime.now(UTC).replace(tzinfo=None)
    stmt = (
        select(Lease)
        .where(Lease.entity_type == entity_type)
        .where(Lease.entity_id == entity_id)
    )
    existing = session.exec(stmt).first()

    if existing is not None:
        if existing.expires_at > now and existing.owner != owner:
            return None  # held by someone else
        session.delete(existing)
        session.flush()

    lease = Lease(
        entity_type=entity_type,
        entity_id=entity_id,
        owner=owner,
        acquired_at=now,
        expires_at=now + timedelta(seconds=ttl_seconds),
    )
    session.add(lease)
    return lease


def release_lease(session: Session, entity_type: str, entity_id: str, owner: str) -> bool:
    """Release a lease. Returns True if it was held by owner and released."""
    stmt = (
        select(Lease)
        .where(Lease.entity_type == entity_type)
        .where(Lease.entity_id == entity_id)
        .where(Lease.owner == owner)
    )
    lease = session.exec(stmt).first()
    if lease is None:
        return False
    session.delete(lease)
    return True


def cleanup_expired_leases(session: Session) -> int:
    """Delete all expired leases. Returns count deleted."""
    now = datetime.now(UTC).replace(tzinfo=None)
    stmt = select(Lease).where(Lease.expires_at <= now)
    expired = list(session.exec(stmt).all())
    for lease in expired:
        session.delete(lease)
    return len(expired)
