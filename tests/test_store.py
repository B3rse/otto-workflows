"""
tests/test_store.py
-------------------
Full test coverage for the store layer.
Uses an in-memory SQLite database — no files written.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event as sa_event
from sqlmodel import create_engine

from otto.store import queries as q
from otto.store.db import get_session, init_db
from otto.store.tables import (
    ArtifactKind,
    EdgeKind,
    EdgeStatus,
    EventEntityType,
    EventType,
    ExternalJobStatus,
    RunStatus,
    TaskRunStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    @sa_event.listens_for(eng, "connect")
    def set_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    init_db(eng)
    return eng


@pytest.fixture()
def session(engine):
    with get_session(engine) as s:
        yield s


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

class TestRun:
    def test_create_and_get(self, session):
        run = q.create_run(session, workflow_name="wf_a", parameters={"foo": 1})
        session.commit()
        fetched = q.get_run(session, run.id)
        assert fetched is not None
        assert fetched.workflow_name == "wf_a"
        assert fetched.parameters == {"foo": 1}
        assert fetched.status == RunStatus.PENDING

    def test_get_runs_filter_by_status(self, session):
        r1 = q.create_run(session, workflow_name="wf_a")
        _r2 = q.create_run(session, workflow_name="wf_b")
        session.commit()
        q.update_run_status(session, r1.id, RunStatus.RUNNING)
        session.commit()
        running = q.get_runs(session, status=RunStatus.RUNNING)
        assert len(running) == 1
        assert running[0].id == r1.id

    def test_update_run_status_with_timestamps(self, session):
        run = q.create_run(session, workflow_name="wf_ts")
        session.commit()
        now = datetime.now(UTC)
        q.update_run_status(session, run.id, RunStatus.SUCCEEDED, finished_at=now)
        session.commit()
        fetched = q.get_run(session, run.id)
        assert fetched.status == RunStatus.SUCCEEDED
        assert fetched.finished_at is not None

    def test_get_run_missing(self, session):
        assert q.get_run(session, "no-such-id") is None


# ---------------------------------------------------------------------------
# TaskRun tests
# ---------------------------------------------------------------------------

class TestTaskRun:
    def test_create_basic(self, session):
        run = q.create_run(session, workflow_name="wf")
        session.commit()
        tr = q.create_task_run(session, run_id=run.id, name="step_a")
        session.commit()
        fetched = q.get_task_run(session, tr.id)
        assert fetched.name == "step_a"
        assert fetched.status == TaskRunStatus.PENDING
        assert fetched.attempt == 0

    def test_create_shard(self, session):
        run = q.create_run(session, workflow_name="wf")
        tr_parent = q.create_task_run(session, run_id=run.id, name="scatter")
        session.commit()
        _shard = q.create_task_run(
            session,
            run_id=run.id,
            name="scatter",
            parent_task_run_id=tr_parent.id,
            shard_index=0,
            shard_key="sample_A",
        )
        session.commit()
        shards = q.get_shards_for_task(session, tr_parent.id)
        assert len(shards) == 1
        assert shards[0].shard_key == "sample_A"

    def test_get_ready_task_runs(self, session):
        run = q.create_run(session, workflow_name="wf")
        tr1 = q.create_task_run(session, run_id=run.id, name="a",
                                 status=TaskRunStatus.READY)
        _tr2 = q.create_task_run(session, run_id=run.id, name="b",
                                 status=TaskRunStatus.PENDING)
        session.commit()
        ready = q.get_ready_task_runs(session, run.id)
        assert len(ready) == 1
        assert ready[0].id == tr1.id

    def test_resource_escalation(self, session):
        profiles = [
            {"cpus": 4, "mem_gb": 16, "walltime": "2:00:00"},
            {"cpus": 8, "mem_gb": 32, "walltime": "4:00:00"},
            {"cpus": 16, "mem_gb": 64, "walltime": "8:00:00"},
        ]
        run = q.create_run(session, workflow_name="wf")
        tr = q.create_task_run(
            session,
            run_id=run.id,
            name="heavy",
            resource_profiles=profiles,
            escalate_on=["OOM", "TIMEOUT"],
        )
        session.commit()

        assert q.get_active_resource_profile(tr) == profiles[0]

        q.increment_task_run_attempt(session, tr.id, escalate_profile=True)
        session.commit()
        tr = q.get_task_run(session, tr.id)
        assert tr.attempt == 1
        assert tr.resource_profile_index == 1
        assert q.get_active_resource_profile(tr) == profiles[1]

    def test_escalation_beyond_max_raises(self, session):
        run = q.create_run(session, workflow_name="wf")
        tr = q.create_task_run(
            session,
            run_id=run.id,
            name="heavy",
            resource_profiles=[{"cpus": 4}],
        )
        session.commit()
        with pytest.raises(ValueError, match="No next resource profile"):
            q.increment_task_run_attempt(session, tr.id, escalate_profile=True)

    def test_no_resource_profiles_returns_none(self, session):
        run = q.create_run(session, workflow_name="wf")
        tr = q.create_task_run(session, run_id=run.id, name="bare")
        assert q.get_active_resource_profile(tr) is None


# ---------------------------------------------------------------------------
# Edge tests
# ---------------------------------------------------------------------------

class TestEdge:
    def test_create_and_resolve(self, session):
        run = q.create_run(session, workflow_name="wf")
        tr_a = q.create_task_run(session, run_id=run.id, name="a")
        tr_b = q.create_task_run(session, run_id=run.id, name="b")
        session.commit()

        edge = q.create_edge(
            session,
            run_id=run.id,
            upstream_task_run_id=tr_a.id,
            downstream_task_run_id=tr_b.id,
            output_name="result",
            input_name="data",
        )
        session.commit()
        assert edge.status == EdgeStatus.PENDING

        q.resolve_edge(session, edge.id)
        session.commit()
        fetched = session.get(q.Edge, edge.id)
        assert fetched.status == EdgeStatus.RESOLVED

    def test_all_upstream_edges_resolved_no_edges(self, session):
        run = q.create_run(session, workflow_name="wf")
        tr = q.create_task_run(session, run_id=run.id, name="root")
        session.commit()
        # No edges → immediately runnable
        assert q.all_upstream_edges_resolved(session, tr.id) is True

    def test_all_upstream_edges_resolved_mixed(self, session):
        run = q.create_run(session, workflow_name="wf")
        tr_a = q.create_task_run(session, run_id=run.id, name="a")
        tr_b = q.create_task_run(session, run_id=run.id, name="b")
        tr_c = q.create_task_run(session, run_id=run.id, name="c")
        session.commit()

        e1 = q.create_edge(session, run_id=run.id,
                            upstream_task_run_id=tr_a.id,
                            downstream_task_run_id=tr_c.id)
        e2 = q.create_edge(session, run_id=run.id,
                            upstream_task_run_id=tr_b.id,
                            downstream_task_run_id=tr_c.id)
        session.commit()

        assert q.all_upstream_edges_resolved(session, tr_c.id) is False

        q.resolve_edge(session, e1.id)
        session.commit()
        assert q.all_upstream_edges_resolved(session, tr_c.id) is False

        q.resolve_edge(session, e2.id)
        session.commit()
        assert q.all_upstream_edges_resolved(session, tr_c.id) is True

    def test_dynamic_edge_kind(self, session):
        run = q.create_run(session, workflow_name="wf")
        tr_a = q.create_task_run(session, run_id=run.id, name="scatter")
        tr_b = q.create_task_run(session, run_id=run.id, name="process",
                                  shard_index=0)
        session.commit()
        edge = q.create_edge(
            session,
            run_id=run.id,
            upstream_task_run_id=tr_a.id,
            downstream_task_run_id=tr_b.id,
            kind=EdgeKind.DYNAMIC,
        )
        session.commit()
        assert edge.kind == EdgeKind.DYNAMIC


# ---------------------------------------------------------------------------
# Artifact tests
# ---------------------------------------------------------------------------

class TestArtifact:
    def test_create_single(self, session):
        run = q.create_run(session, workflow_name="wf")
        tr = q.create_task_run(session, run_id=run.id, name="a")
        session.commit()
        _art = q.create_artifact(
            session,
            run_id=run.id,
            produced_by_task_run_id=tr.id,
            name="output_bam",
            uri="s3://bucket/sample.bam",
        )
        session.commit()
        arts = q.get_artifacts_for_task_run(session, tr.id)
        assert len(arts) == 1
        assert arts[0].uri == "s3://bucket/sample.bam"

    def test_create_array_artifact(self, session):
        run = q.create_run(session, workflow_name="wf")
        tr = q.create_task_run(session, run_id=run.id, name="scatter")
        session.commit()

        uris = ["s3://bucket/a.bam", "s3://bucket/b.bam", "s3://bucket/c.bam"]
        parent, children = q.create_array_artifact(
            session,
            run_id=run.id,
            produced_by_task_run_id=tr.id,
            name="sharded_bams",
            elements=uris,
        )
        session.commit()

        assert parent.kind == ArtifactKind.ARRAY
        assert parent.uri is None
        assert len(children) == 3
        assert children[0].array_index == 0
        assert children[2].uri == uris[2]

        fetched_children = q.get_array_elements(session, parent.id)
        assert [c.array_index for c in fetched_children] == [0, 1, 2]

    def test_get_artifact_by_name(self, session):
        run = q.create_run(session, workflow_name="wf")
        tr = q.create_task_run(session, run_id=run.id, name="a")
        session.commit()
        q.create_artifact(session, run_id=run.id,
                          produced_by_task_run_id=tr.id,
                          name="result", uri="/out/result.txt")
        session.commit()
        art = q.get_artifact_by_name(session, tr.id, "result")
        assert art is not None
        assert art.uri == "/out/result.txt"

        missing = q.get_artifact_by_name(session, tr.id, "nonexistent")
        assert missing is None


# ---------------------------------------------------------------------------
# ExternalJob tests
# ---------------------------------------------------------------------------

class TestExternalJob:
    def test_create_and_update(self, session):
        run = q.create_run(session, workflow_name="wf")
        tr = q.create_task_run(session, run_id=run.id, name="a")
        session.commit()

        job = q.create_external_job(
            session,
            task_run_id=tr.id,
            run_id=run.id,
            backend="slurm",
            ext_id="12345",
            attempt=0,
            resource_profile={"cpus": 4, "mem_gb": 16},
        )
        session.commit()
        assert job.status == ExternalJobStatus.SUBMITTED

        q.update_external_job(
            session,
            job.id,
            status=ExternalJobStatus.SUCCEEDED,
            exit_code=0,
            finished_at=datetime.now(UTC),
        )
        session.commit()
        fetched = session.get(q.ExternalJob, job.id)
        assert fetched.status == ExternalJobStatus.SUCCEEDED
        assert fetched.exit_code == 0

    def test_get_active_external_jobs(self, session):
        run = q.create_run(session, workflow_name="wf")
        tr = q.create_task_run(session, run_id=run.id, name="a")
        session.commit()

        j1 = q.create_external_job(session, task_run_id=tr.id,
                                    run_id=run.id, backend="slurm",
                                    ext_id="100")
        _j2 = q.create_external_job(session, task_run_id=tr.id,
                                    run_id=run.id, backend="slurm",
                                    ext_id="101")
        session.commit()

        q.update_external_job(session, j1.id,
                              status=ExternalJobStatus.SUCCEEDED)
        session.commit()

        active = q.get_active_external_jobs(session, run.id)
        assert len(active) == 1
        assert active[0].ext_id == "101"


# ---------------------------------------------------------------------------
# Event log tests
# ---------------------------------------------------------------------------

class TestEvents:
    def test_emit_and_retrieve(self, session):
        run = q.create_run(session, workflow_name="wf")
        session.commit()
        q.emit_event(
            session,
            entity_type=EventEntityType.RUN,
            entity_id=run.id,
            run_id=run.id,
            event_type=EventType.RUN_CREATED,
            payload={"source": "test"},
        )
        session.commit()
        events = q.get_events(session, run_id=run.id)
        assert len(events) == 1
        assert events[0].event_type == EventType.RUN_CREATED
        assert events[0].payload["source"] == "test"

    def test_filter_by_event_type(self, session):
        run = q.create_run(session, workflow_name="wf")
        session.commit()
        q.emit_event(session, entity_type=EventEntityType.RUN,
                     entity_id=run.id, run_id=run.id,
                     event_type=EventType.RUN_CREATED)
        q.emit_event(session, entity_type=EventEntityType.RUN,
                     entity_id=run.id, run_id=run.id,
                     event_type=EventType.RUN_STARTED)
        session.commit()
        started = q.get_events(session, event_type=EventType.RUN_STARTED)
        assert len(started) == 1


# ---------------------------------------------------------------------------
# Lease tests
# ---------------------------------------------------------------------------

class TestLeases:
    def test_acquire_and_release(self, session):
        lease = q.acquire_lease(
            session,
            entity_type="task_run",
            entity_id="tr-001",
            owner="loop:pid:1234",
            ttl_seconds=60,
        )
        session.commit()
        assert lease is not None

        # Same owner can re-acquire (idempotent)
        lease2 = q.acquire_lease(
            session,
            entity_type="task_run",
            entity_id="tr-001",
            owner="loop:pid:1234",
            ttl_seconds=60,
        )
        session.commit()
        assert lease2 is not None

        released = q.release_lease(session, "task_run", "tr-001", "loop:pid:1234")
        session.commit()
        assert released is True

    def test_acquire_blocked_by_another_owner(self, session):
        q.acquire_lease(session, entity_type="task_run", entity_id="tr-002",
                        owner="owner_A", ttl_seconds=300)
        session.commit()

        blocked = q.acquire_lease(session, entity_type="task_run", entity_id="tr-002",
                                   owner="owner_B", ttl_seconds=300)
        assert blocked is None

    def test_expired_lease_can_be_taken(self, session):
        lease = q.acquire_lease(session, entity_type="task_run", entity_id="tr-003",
                                  owner="old_owner", ttl_seconds=1)
        session.commit()
        # Manually expire it
        lease.expires_at = datetime.now(UTC) - timedelta(seconds=10)
        session.add(lease)
        session.commit()

        new_lease = q.acquire_lease(session, entity_type="task_run", entity_id="tr-003",
                                     owner="new_owner", ttl_seconds=60)
        session.commit()
        assert new_lease is not None
        assert new_lease.owner == "new_owner"

    def test_cleanup_expired(self, session):
        lease = q.acquire_lease(session, entity_type="run", entity_id="r-001",
                                  owner="loop", ttl_seconds=1)
        session.commit()
        lease.expires_at = datetime.now(UTC) - timedelta(seconds=5)
        session.add(lease)
        session.commit()

        count = q.cleanup_expired_leases(session)
        session.commit()
        assert count == 1
