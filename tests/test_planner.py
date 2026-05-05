"""
tests/test_planner.py
---------------------
Full coverage for the planner layer.

Tests create an in-memory SQLite DB, call plan_run(), and assert that the
correct Run / TaskRun / Edge / Event rows exist with the right field values.
No backends or executors are involved — this is a pure DB-state test.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import event as sa_event
from sqlmodel import create_engine

from otto.model import (
    FileInput,
    FileSpec,
    FolderInput,
    OutputSpec,
    ParameterInput,
    ParameterSpec,
    ResourceProfile,
    ResourceSpec,
    TaskFileInput,
    TaskFolderInput,
    TaskParameterInput,
    TaskTemplate,
    WorkflowTemplate,
)
from otto.planner import PlanResult, plan_run
from otto.store import (
    EdgeKind,
    EdgeStatus,
    EventEntityType,
    EventType,
    RunStatus,
    TaskRunStatus,
    get_events,
    get_task_run,
    init_db,
)
from otto.store.db import get_session

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
# Helpers
# ---------------------------------------------------------------------------

def make_task(**kwargs) -> TaskTemplate:
    defaults: dict[str, Any] = dict(name="step", engine="shell", backend="local")
    return TaskTemplate(**(defaults | kwargs))


def make_workflow(**kwargs) -> WorkflowTemplate:
    defaults: dict[str, Any] = dict(name="wf", tasks=[])
    return WorkflowTemplate(**(defaults | kwargs))


def linear_workflow() -> WorkflowTemplate:
    """A → B linear chain."""
    return make_workflow(
        name="linear",
        tasks=[
            make_task(
                name="a",
                outputs={"result": OutputSpec(type="file")},
            ),
            make_task(
                name="b",
                dependencies=["a"],
                inputs={"x": TaskFileInput(task="a", file="result")},
            ),
        ],
    )


def diamond_workflow() -> WorkflowTemplate:
    """A → B, A → C, B + C → D."""
    return make_workflow(
        name="diamond",
        tasks=[
            make_task(name="a", outputs={"val": OutputSpec(type="value")}),
            make_task(
                name="b",
                dependencies=["a"],
                inputs={"v": TaskParameterInput(task="a", parameter="val")},
                outputs={"out": OutputSpec(type="file")},
            ),
            make_task(
                name="c",
                dependencies=["a"],
                inputs={"v": TaskParameterInput(task="a", parameter="val")},
                outputs={"out": OutputSpec(type="file")},
            ),
            make_task(
                name="d",
                dependencies=["b", "c"],
                inputs={
                    "b_out": TaskFileInput(task="b", file="out"),
                    "c_out": TaskFileInput(task="c", file="out"),
                },
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Basic creation
# ---------------------------------------------------------------------------

class TestPlanRunBasic:

    def test_run_created(self, session):
        wf = make_workflow(name="my_wf")
        result = plan_run(session, wf)
        session.commit()
        assert result.run is not None
        assert result.run.workflow_name == "my_wf"
        assert result.run.status == RunStatus.PENDING

    def test_run_version_stored(self, session):
        wf = make_workflow(name="versioned", version="2.1.0")
        result = plan_run(session, wf)
        session.commit()
        assert result.run.workflow_version == "2.1.0"

    def test_run_label_stored(self, session):
        wf = make_workflow()
        result = plan_run(session, wf, label="batch-42")
        session.commit()
        assert result.run.label == "batch-42"

    def test_run_parameters_stored(self, session):
        wf = make_workflow(inputs={"sample": ParameterSpec(type="str", required=True)})
        params = {"sample": "S001"}
        result = plan_run(session, wf, parameters=params)
        session.commit()
        assert result.run.parameters == params

    def test_empty_workflow(self, session):
        wf = make_workflow()
        result = plan_run(session, wf)
        session.commit()
        assert result.run is not None
        assert result.task_runs == {}
        assert result.edges == []

    def test_linear_workflow_creates_correct_records(self, session):
        result = plan_run(session, linear_workflow())
        session.commit()
        assert len(result.task_runs) == 2
        assert "a" in result.task_runs
        assert "b" in result.task_runs
        assert len(result.edges) == 1

    def test_diamond_workflow_creates_correct_records(self, session):
        result = plan_run(session, diamond_workflow())
        session.commit()
        assert len(result.task_runs) == 4
        assert len(result.edges) == 4  # A→B, A→C, B→D, C→D

    def test_result_type(self, session):
        result = plan_run(session, make_workflow())
        assert isinstance(result, PlanResult)

    def test_caller_commits_and_data_persists(self, engine):
        wf = linear_workflow()
        with get_session(engine) as s:
            result = plan_run(s, wf)
            run_id = result.run.id
            tr_a_id = result.task_runs["a"].id
            s.commit()

        with get_session(engine) as s:
            from otto.store import get_run
            run = get_run(s, run_id)
            assert run is not None
            assert run.workflow_name == "linear"
            tr = get_task_run(s, tr_a_id)
            assert tr is not None
            assert tr.name == "a"


# ---------------------------------------------------------------------------
# TaskRun status
# ---------------------------------------------------------------------------

class TestTaskRunStatus:

    def test_root_task_is_ready(self, session):
        wf = make_workflow(tasks=[make_task(name="root")])
        result = plan_run(session, wf)
        session.commit()
        assert result.task_runs["root"].status == TaskRunStatus.READY

    def test_dependent_task_is_pending(self, session):
        result = plan_run(session, linear_workflow())
        session.commit()
        assert result.task_runs["a"].status == TaskRunStatus.READY
        assert result.task_runs["b"].status == TaskRunStatus.PENDING

    def test_multiple_roots_all_ready(self, session):
        wf = make_workflow(
            tasks=[
                make_task(name="a"),
                make_task(name="b"),
                make_task(name="c"),
            ]
        )
        result = plan_run(session, wf)
        session.commit()
        for name in ("a", "b", "c"):
            assert result.task_runs[name].status == TaskRunStatus.READY

    def test_diamond_statuses(self, session):
        result = plan_run(session, diamond_workflow())
        session.commit()
        assert result.task_runs["a"].status == TaskRunStatus.READY
        assert result.task_runs["b"].status == TaskRunStatus.PENDING
        assert result.task_runs["c"].status == TaskRunStatus.PENDING
        assert result.task_runs["d"].status == TaskRunStatus.PENDING


# ---------------------------------------------------------------------------
# TaskRun core fields
# ---------------------------------------------------------------------------

class TestTaskRunFields:

    def test_engine_and_backend_stored(self, session):
        wf = make_workflow(
            tasks=[make_task(name="t", engine="nextflow", backend="slurm")]
        )
        result = plan_run(session, wf)
        session.commit()
        tr = result.task_runs["t"]
        assert tr.engine == "nextflow"
        assert tr.backend == "slurm"

    def test_cmd_stored(self, session):
        wf = make_workflow(
            tasks=[make_task(name="t", cmd="echo hello > {outputs.out}")]
        )
        result = plan_run(session, wf)
        session.commit()
        assert result.task_runs["t"].cmd == "echo hello > {outputs.out}"

    def test_cmd_none_when_not_set(self, session):
        wf = make_workflow(tasks=[make_task(name="t")])
        result = plan_run(session, wf)
        session.commit()
        assert result.task_runs["t"].cmd is None

    def test_run_id_on_task_run(self, session):
        result = plan_run(session, linear_workflow())
        session.commit()
        for tr in result.task_runs.values():
            assert tr.run_id == result.run.id

    def test_attempt_starts_at_zero(self, session):
        result = plan_run(session, make_workflow(tasks=[make_task(name="t")]))
        session.commit()
        assert result.task_runs["t"].attempt == 0

    def test_resource_profile_index_starts_at_zero(self, session):
        result = plan_run(session, make_workflow(tasks=[make_task(name="t")]))
        session.commit()
        assert result.task_runs["t"].resource_profile_index == 0

    def test_shard_fields_not_set_for_regular_task(self, session):
        result = plan_run(session, make_workflow(tasks=[make_task(name="t")]))
        session.commit()
        tr = result.task_runs["t"]
        assert tr.parent_task_run_id is None
        assert tr.shard_index is None
        assert tr.shard_key is None


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

class TestEdges:

    def test_edge_kind_is_static(self, session):
        result = plan_run(session, linear_workflow())
        session.commit()
        for edge in result.edges:
            assert edge.kind == EdgeKind.STATIC

    def test_edge_status_is_pending(self, session):
        result = plan_run(session, linear_workflow())
        session.commit()
        for edge in result.edges:
            assert edge.status == EdgeStatus.PENDING

    def test_edge_run_id(self, session):
        result = plan_run(session, linear_workflow())
        session.commit()
        for edge in result.edges:
            assert edge.run_id == result.run.id

    def test_linear_edge_direction(self, session):
        result = plan_run(session, linear_workflow())
        session.commit()
        edge = result.edges[0]
        assert edge.upstream_task_run_id == result.task_runs["a"].id
        assert edge.downstream_task_run_id == result.task_runs["b"].id

    def test_diamond_edge_count(self, session):
        result = plan_run(session, diamond_workflow())
        session.commit()
        assert len(result.edges) == 4

    def test_diamond_edge_directions(self, session):
        result = plan_run(session, diamond_workflow())
        session.commit()
        trs = result.task_runs

        upstream_downstream = {
            (e.upstream_task_run_id, e.downstream_task_run_id)
            for e in result.edges
        }
        expected = {
            (trs["a"].id, trs["b"].id),
            (trs["a"].id, trs["c"].id),
            (trs["b"].id, trs["d"].id),
            (trs["c"].id, trs["d"].id),
        }
        assert upstream_downstream == expected

    def test_no_edges_for_independent_tasks(self, session):
        wf = make_workflow(tasks=[make_task(name="a"), make_task(name="b")])
        result = plan_run(session, wf)
        session.commit()
        assert result.edges == []

    def test_chain_of_three_has_two_edges(self, session):
        wf = make_workflow(
            tasks=[
                make_task(name="a", outputs={"o": OutputSpec(type="file")}),
                make_task(
                    name="b",
                    dependencies=["a"],
                    inputs={"i": TaskFileInput(task="a", file="o")},
                    outputs={"o": OutputSpec(type="file")},
                ),
                make_task(
                    name="c",
                    dependencies=["b"],
                    inputs={"i": TaskFileInput(task="b", file="o")},
                ),
            ]
        )
        result = plan_run(session, wf)
        session.commit()
        assert len(result.edges) == 2


# ---------------------------------------------------------------------------
# Input wiring serialization
# ---------------------------------------------------------------------------

class TestInputsSnapshot:

    def test_parameter_input_serialized(self, session):
        wf = make_workflow(
            inputs={"genome": ParameterSpec(type="str", default="hg38")},
            tasks=[
                make_task(
                    name="t",
                    inputs={"ref": ParameterInput(parameter="genome")},
                )
            ],
        )
        result = plan_run(session, wf)
        session.commit()
        tr_inputs = result.task_runs["t"].inputs
        assert tr_inputs is not None
        assert tr_inputs["ref"] == {"parameter": "genome"}

    def test_file_input_serialized(self, session):
        wf = make_workflow(
            inputs={"bam": FileSpec(type="file")},
            tasks=[
                make_task(
                    name="t",
                    inputs={"input_bam": FileInput(file="bam")},
                )
            ],
        )
        result = plan_run(session, wf)
        session.commit()
        tr_inputs = result.task_runs["t"].inputs
        assert tr_inputs["input_bam"] == {"file": "bam"}

    def test_folder_input_serialized(self, session):
        wf = make_workflow(
            inputs={"ref_dir": FileSpec(type="directory")},
            tasks=[
                make_task(
                    name="t",
                    inputs={"ref": FolderInput(folder="ref_dir")},
                )
            ],
        )
        result = plan_run(session, wf)
        session.commit()
        tr_inputs = result.task_runs["t"].inputs
        assert tr_inputs["ref"] == {"folder": "ref_dir"}

    def test_task_file_input_serialized(self, session):
        result = plan_run(session, linear_workflow())
        session.commit()
        tr_b_inputs = result.task_runs["b"].inputs
        assert tr_b_inputs is not None
        assert tr_b_inputs["x"] == {"task": "a", "file": "result"}

    def test_task_parameter_input_serialized(self, session):
        result = plan_run(session, diamond_workflow())
        session.commit()
        tr_b_inputs = result.task_runs["b"].inputs
        assert tr_b_inputs["v"] == {"task": "a", "parameter": "val"}

    def test_task_folder_input_serialized(self, session):
        wf = make_workflow(
            tasks=[
                make_task(name="a", outputs={"refs": OutputSpec(type="directory")}),
                make_task(
                    name="b",
                    dependencies=["a"],
                    inputs={"r": TaskFolderInput(task="a", folder="refs")},
                ),
            ]
        )
        result = plan_run(session, wf)
        session.commit()
        tr_b_inputs = result.task_runs["b"].inputs
        assert tr_b_inputs["r"] == {"task": "a", "folder": "refs"}

    def test_no_inputs_stored_as_empty_dict(self, session):
        wf = make_workflow(tasks=[make_task(name="t")])
        result = plan_run(session, wf)
        session.commit()
        tr = result.task_runs["t"]
        # planner passes None (empty snapshot is falsy); create_task_run
        # normalises None → {}, so the stored value is always an empty dict
        assert tr.inputs == {}

    def test_inputs_round_trip_from_db(self, engine):
        """Inputs JSON round-trips correctly through SQLite (no silent data loss)."""
        wf = make_workflow(
            inputs={
                "genome": ParameterSpec(type="str", default="hg38"),
                "reads":  FileSpec(type="file[]"),
            },
            tasks=[
                make_task(
                    name="t",
                    inputs={
                        "ref": ParameterInput(parameter="genome"),
                        "fq":  FileInput(file="reads"),
                    },
                )
            ],
        )
        tr_id: str
        with get_session(engine) as s:
            result = plan_run(s, wf)
            tr_id = result.task_runs["t"].id
            s.commit()

        with get_session(engine) as s:
            tr = get_task_run(s, tr_id)
            assert tr is not None
            assert tr.inputs == {
                "ref": {"parameter": "genome"},
                "fq":  {"file": "reads"},
            }

    def test_outputs_round_trip_from_db(self, engine):
        """Outputs JSON round-trips correctly through SQLite."""
        wf = make_workflow(
            tasks=[
                make_task(
                    name="t",
                    outputs={
                        "bam":   OutputSpec(type="file", pattern="*.bam"),
                        "count": OutputSpec(type="value"),
                    },
                )
            ]
        )
        tr_id: str
        with get_session(engine) as s:
            result = plan_run(s, wf)
            tr_id = result.task_runs["t"].id
            s.commit()

        with get_session(engine) as s:
            tr = get_task_run(s, tr_id)
            assert tr is not None
            assert tr.outputs == {
                "bam":   {"type": "file", "pattern": "*.bam"},
                "count": {"type": "value", "pattern": None},
            }

    def test_multiple_inputs_all_serialized(self, session):
        wf = make_workflow(
            inputs={
                "genome": ParameterSpec(type="str", default="hg38"),
                "bam":    FileSpec(type="file"),
            },
            tasks=[
                make_task(
                    name="t",
                    inputs={
                        "ref": ParameterInput(parameter="genome"),
                        "input_bam": FileInput(file="bam"),
                    },
                )
            ],
        )
        result = plan_run(session, wf)
        session.commit()
        tr_inputs = result.task_runs["t"].inputs
        assert tr_inputs is not None
        assert len(tr_inputs) == 2
        assert "ref" in tr_inputs
        assert "input_bam" in tr_inputs


# ---------------------------------------------------------------------------
# Output spec serialization
# ---------------------------------------------------------------------------

class TestOutputsSnapshot:

    def test_file_output_serialized(self, session):
        wf = make_workflow(
            tasks=[
                make_task(
                    name="t",
                    outputs={"bam": OutputSpec(type="file", pattern="*.bam")},
                )
            ]
        )
        result = plan_run(session, wf)
        session.commit()
        tr_outputs = result.task_runs["t"].outputs
        assert tr_outputs is not None
        assert tr_outputs["bam"]["type"] == "file"
        assert tr_outputs["bam"]["pattern"] == "*.bam"

    def test_value_output_serialized(self, session):
        wf = make_workflow(
            tasks=[
                make_task(
                    name="t",
                    outputs={"count": OutputSpec(type="value")},
                )
            ]
        )
        result = plan_run(session, wf)
        session.commit()
        tr_outputs = result.task_runs["t"].outputs
        assert tr_outputs["count"]["type"] == "value"

    def test_array_output_serialized(self, session):
        wf = make_workflow(
            tasks=[
                make_task(
                    name="t",
                    outputs={"bams": OutputSpec(type="file[]")},
                )
            ]
        )
        result = plan_run(session, wf)
        session.commit()
        tr_outputs = result.task_runs["t"].outputs
        assert tr_outputs["bams"]["type"] == "file[]"

    def test_directory_output_serialized(self, session):
        wf = make_workflow(
            tasks=[
                make_task(
                    name="t",
                    outputs={"ref_dir": OutputSpec(type="directory")},
                )
            ]
        )
        result = plan_run(session, wf)
        session.commit()
        tr_outputs = result.task_runs["t"].outputs
        assert tr_outputs["ref_dir"]["type"] == "directory"

    def test_no_outputs_stored_as_none(self, session):
        wf = make_workflow(tasks=[make_task(name="t")])
        result = plan_run(session, wf)
        session.commit()
        assert result.task_runs["t"].outputs is None

    def test_multiple_outputs_all_serialized(self, session):
        wf = make_workflow(
            tasks=[
                make_task(
                    name="t",
                    outputs={
                        "bam":   OutputSpec(type="file"),
                        "log":   OutputSpec(type="value"),
                        "index": OutputSpec(type="file", pattern="*.bai"),
                    },
                )
            ]
        )
        result = plan_run(session, wf)
        session.commit()
        tr_outputs = result.task_runs["t"].outputs
        assert tr_outputs is not None
        assert len(tr_outputs) == 3


# ---------------------------------------------------------------------------
# Resource profiles
# ---------------------------------------------------------------------------

class TestResourceProfiles:

    def test_resource_profiles_serialized(self, session):
        profiles = [
            ResourceProfile(cpus=4, mem_gb=16.0, walltime="2:00:00"),
            ResourceProfile(cpus=8, mem_gb=32.0, walltime="4:00:00"),
        ]
        wf = make_workflow(
            tasks=[
                make_task(
                    name="t",
                    resources=ResourceSpec(
                        profiles=profiles,
                        escalate_on=["OOM", "TIMEOUT"],
                    ),
                )
            ]
        )
        result = plan_run(session, wf)
        session.commit()
        tr = result.task_runs["t"]
        assert tr.resource_profiles is not None
        assert len(tr.resource_profiles) == 2
        assert tr.resource_profiles[0]["cpus"] == 4
        assert tr.resource_profiles[1]["mem_gb"] == 32.0

    def test_escalate_on_stored(self, session):
        wf = make_workflow(
            tasks=[
                make_task(
                    name="t",
                    resources=ResourceSpec(
                        profiles=[ResourceProfile()],
                        escalate_on=["OOM"],
                    ),
                )
            ]
        )
        result = plan_run(session, wf)
        session.commit()
        assert result.task_runs["t"].escalate_on == ["OOM"]

    def test_no_resource_profiles_stored_as_empty_list(self, session):
        wf = make_workflow(tasks=[make_task(name="t")])
        result = plan_run(session, wf)
        session.commit()
        assert result.task_runs["t"].resource_profiles == []

    def test_no_escalate_on_stored_as_none(self, session):
        wf = make_workflow(tasks=[make_task(name="t")])
        result = plan_run(session, wf)
        session.commit()
        assert result.task_runs["t"].escalate_on is None

    def test_resource_profile_extra_fields_serialized(self, session):
        wf = make_workflow(
            tasks=[
                make_task(
                    name="t",
                    resources=ResourceSpec(
                        profiles=[
                            ResourceProfile(
                                cpus=2, mem_gb=8.0,
                                extra={"partition": "gpu", "gpus": 1},
                            )
                        ],
                    ),
                )
            ]
        )
        result = plan_run(session, wf)
        session.commit()
        stored = result.task_runs["t"].resource_profiles[0]
        assert stored["extra"]["partition"] == "gpu"
        assert stored["extra"]["gpus"] == 1


# ---------------------------------------------------------------------------
# Scatter
# ---------------------------------------------------------------------------

class TestScatter:

    def test_scatter_field_stored(self, session):
        wf = make_workflow(
            inputs={"reads": FileSpec(type="file[]", required=True)},
            tasks=[
                make_task(
                    name="t",
                    scatter="fq",
                    inputs={"fq": FileInput(file="reads")},
                    outputs={"result": OutputSpec(type="file")},
                )
            ],
        )
        result = plan_run(session, wf, parameters={"reads": "/data/reads"})
        session.commit()
        tr = result.task_runs["t"]
        assert tr.scatter == "fq"

    def test_scatter_method_flat_stored(self, session):
        wf = make_workflow(
            inputs={"reads": FileSpec(type="file[]", required=True)},
            tasks=[
                make_task(
                    name="t",
                    scatter="fq",
                    scatter_method="flat",
                    inputs={"fq": FileInput(file="reads")},
                )
            ],
        )
        result = plan_run(session, wf, parameters={"reads": "/data"})
        session.commit()
        assert result.task_runs["t"].scatter_method == "flat"

    def test_scatter_method_grouped_stored(self, session):
        wf = make_workflow(
            inputs={"reads": FileSpec(type="file[]", required=True)},
            tasks=[
                make_task(
                    name="t",
                    scatter="fq",
                    scatter_method="grouped",
                    inputs={"fq": FileInput(file="reads")},
                )
            ],
        )
        result = plan_run(session, wf, parameters={"reads": "/data"})
        session.commit()
        assert result.task_runs["t"].scatter_method == "grouped"

    def test_no_scatter_stored_as_none(self, session):
        wf = make_workflow(tasks=[make_task(name="t")])
        result = plan_run(session, wf)
        session.commit()
        tr = result.task_runs["t"]
        assert tr.scatter is None
        assert tr.scatter_method is None

    def test_scatter_task_with_upstream_dependency(self, session):
        """Scatter task that fans over an upstream task's array output."""
        wf = make_workflow(
            tasks=[
                make_task(
                    name="produce",
                    outputs={"bams": OutputSpec(type="file[]")},
                ),
                make_task(
                    name="process",
                    dependencies=["produce"],
                    scatter="bam",
                    inputs={"bam": TaskFileInput(task="produce", file="bams")},
                    outputs={"result": OutputSpec(type="file")},
                ),
            ]
        )
        result = plan_run(session, wf)
        session.commit()
        tr = result.task_runs["process"]
        assert tr.scatter == "bam"
        assert tr.status == TaskRunStatus.PENDING  # has a dependency
        assert tr.scatter_method == "flat"

    def test_scatter_root_task_is_ready(self, session):
        """Scatter task with no dependencies (scatter from workflow input) is READY."""
        wf = make_workflow(
            inputs={"reads": FileSpec(type="file[]", required=True)},
            tasks=[
                make_task(
                    name="process",
                    scatter="fq",
                    inputs={"fq": FileInput(file="reads")},
                )
            ],
        )
        result = plan_run(session, wf, parameters={"reads": "/data"})
        session.commit()
        assert result.task_runs["process"].status == TaskRunStatus.READY


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class TestEvents:

    def test_run_created_event_emitted(self, session):
        result = plan_run(session, make_workflow())
        session.commit()
        events = get_events(session, run_id=result.run.id,
                            event_type=EventType.RUN_CREATED)
        assert len(events) == 1
        assert events[0].entity_type == EventEntityType.RUN
        assert events[0].entity_id == result.run.id

    def test_task_created_events_emitted(self, session):
        result = plan_run(session, linear_workflow())
        session.commit()
        events = get_events(session, run_id=result.run.id,
                            event_type=EventType.TASK_CREATED)
        assert len(events) == 2  # one per task

    def test_task_created_event_entity_ids(self, session):
        result = plan_run(session, linear_workflow())
        session.commit()
        events = get_events(session, run_id=result.run.id,
                            event_type=EventType.TASK_CREATED)
        event_entity_ids = {e.entity_id for e in events}
        task_run_ids = {tr.id for tr in result.task_runs.values()}
        assert event_entity_ids == task_run_ids

    def test_event_count_matches_tasks(self, session):
        result = plan_run(session, diamond_workflow())
        session.commit()
        all_events = get_events(session, run_id=result.run.id)
        # 1 RUN_CREATED + 4 TASK_CREATED
        assert len(all_events) == 5

    def test_empty_workflow_emits_run_created_only(self, session):
        result = plan_run(session, make_workflow())
        session.commit()
        all_events = get_events(session, run_id=result.run.id)
        assert len(all_events) == 1
        assert all_events[0].event_type == EventType.RUN_CREATED

    def test_all_events_have_correct_run_id(self, session):
        result = plan_run(session, diamond_workflow())
        session.commit()
        all_events = get_events(session, run_id=result.run.id)
        for ev in all_events:
            assert ev.run_id == result.run.id


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

class TestParameterValidation:

    def test_required_parameter_missing_raises(self, session):
        wf = make_workflow(inputs={"sample": ParameterSpec(required=True)})
        with pytest.raises(ValueError, match="Required inputs not supplied"):
            plan_run(session, wf, parameters={})

    def test_required_file_input_missing_raises(self, session):
        wf = make_workflow(inputs={"bam": FileSpec(required=True)})
        with pytest.raises(ValueError, match="Required inputs not supplied"):
            plan_run(session, wf)

    def test_unknown_parameter_raises(self, session):
        wf = make_workflow(inputs={"sample": ParameterSpec(type="str")})
        with pytest.raises(ValueError, match="Unknown parameters"):
            plan_run(session, wf, parameters={"sample": "S1", "extra": "oops"})

    def test_completely_unknown_parameter_raises(self, session):
        wf = make_workflow()
        with pytest.raises(ValueError, match="Unknown parameters"):
            plan_run(session, wf, parameters={"ghost": "value"})

    def test_optional_parameter_can_be_omitted(self, session):
        wf = make_workflow(
            inputs={"genome": ParameterSpec(type="str", default="hg38")}
        )
        result = plan_run(session, wf)
        session.commit()
        assert result.run is not None

    def test_required_parameter_supplied(self, session):
        wf = make_workflow(
            inputs={
                "sample": ParameterSpec(required=True),
                "genome": ParameterSpec(default="hg38"),
            }
        )
        result = plan_run(session, wf, parameters={"sample": "S001"})
        session.commit()
        assert result.run.parameters["sample"] == "S001"

    def test_no_parameters_no_inputs_ok(self, session):
        wf = make_workflow()
        result = plan_run(session, wf)
        session.commit()
        assert result.run.parameters == {}

    def test_multiple_required_inputs_all_missing_lists_them(self, session):
        wf = make_workflow(
            inputs={
                "sample": ParameterSpec(required=True),
                "bam":    FileSpec(required=True),
            }
        )
        with pytest.raises(ValueError, match="Required inputs not supplied"):
            plan_run(session, wf, parameters={})

    def test_error_before_any_db_write(self, session):
        """Validation error must not leave partial records in the DB."""
        wf = make_workflow(
            inputs={"sample": ParameterSpec(required=True)},
            tasks=[make_task(name="t")],
        )
        with pytest.raises(ValueError):
            plan_run(session, wf, parameters={})
        session.commit()
        # Nothing should have been written
        from otto.store import get_runs
        runs = get_runs(session)
        assert runs == []


# ---------------------------------------------------------------------------
# Multiple plans (isolation)
# ---------------------------------------------------------------------------

class TestMultiplePlans:

    def test_two_plans_are_independent(self, session):
        wf = linear_workflow()
        result1 = plan_run(session, wf, label="run-1")
        result2 = plan_run(session, wf, label="run-2")
        session.commit()

        assert result1.run.id != result2.run.id
        for name in ("a", "b"):
            assert result1.task_runs[name].id != result2.task_runs[name].id

    def test_each_plan_has_own_edges(self, session):
        wf = linear_workflow()
        r1 = plan_run(session, wf)
        r2 = plan_run(session, wf)
        session.commit()

        assert len(r1.edges) == 1
        assert len(r2.edges) == 1
        assert r1.edges[0].id != r2.edges[0].id

    def test_each_plan_has_own_events(self, session):
        wf = make_workflow(tasks=[make_task(name="t")])
        r1 = plan_run(session, wf)
        r2 = plan_run(session, wf)
        session.commit()

        ev1 = get_events(session, run_id=r1.run.id)
        ev2 = get_events(session, run_id=r2.run.id)
        # 1 RUN_CREATED + 1 TASK_CREATED each
        assert len(ev1) == 2
        assert len(ev2) == 2


# ---------------------------------------------------------------------------
# Full realistic workflow round-trip
# ---------------------------------------------------------------------------

class TestRealisticWorkflow:

    def test_rna_seq_like_workflow(self, session):
        """
        Simulates a small RNA-seq pipeline:
          qc → align → sort → index
                              align → count (two inputs from align)
        """
        wf = make_workflow(
            name="rna_seq",
            inputs={
                "sample":  ParameterSpec(type="str", required=True),
                "fastq":   FileSpec(type="file[]", required=True),
                "ref_dir": FileSpec(type="directory"),
            },
            tasks=[
                make_task(
                    name="qc",
                    inputs={"fq": FileInput(file="fastq")},
                    outputs={"report": OutputSpec(type="value")},
                    resources=ResourceSpec(
                        profiles=[ResourceProfile(cpus=2, mem_gb=4.0)],
                    ),
                ),
                make_task(
                    name="align",
                    dependencies=["qc"],
                    inputs={
                        "fq":  FileInput(file="fastq"),
                        "ref": FolderInput(folder="ref_dir"),
                        "qc_report": TaskParameterInput(task="qc", parameter="report"),
                    },
                    outputs={
                        "bam":    OutputSpec(type="file", pattern="*.bam"),
                        "log":    OutputSpec(type="value"),
                    },
                    resources=ResourceSpec(
                        profiles=[
                            ResourceProfile(cpus=8, mem_gb=32.0, walltime="4:00:00"),
                            ResourceProfile(cpus=16, mem_gb=64.0, walltime="8:00:00"),
                        ],
                        escalate_on=["OOM", "TIMEOUT"],
                    ),
                    cmd="bwa mem ...",
                ),
                make_task(
                    name="sort",
                    dependencies=["align"],
                    inputs={"bam": TaskFileInput(task="align", file="bam")},
                    outputs={"sorted_bam": OutputSpec(type="file")},
                ),
                make_task(
                    name="index",
                    dependencies=["sort"],
                    inputs={"bam": TaskFileInput(task="sort", file="sorted_bam")},
                    outputs={"bai": OutputSpec(type="file")},
                ),
                make_task(
                    name="count",
                    dependencies=["align"],
                    inputs={
                        "bam":    TaskFileInput(task="align", file="bam"),
                        "sample": ParameterInput(parameter="sample"),
                    },
                    outputs={"counts": OutputSpec(type="value[]")},
                ),
            ],
        )

        result = plan_run(
            session, wf,
            parameters={"sample": "S001", "fastq": "/data/S001.fastq.gz"},
            label="test-run",
        )
        session.commit()

        # Run
        assert result.run.workflow_name == "rna_seq"
        assert result.run.label == "test-run"
        assert result.run.parameters["sample"] == "S001"

        # TaskRun count
        assert len(result.task_runs) == 5

        # Statuses
        assert result.task_runs["qc"].status == TaskRunStatus.READY
        for name in ("align", "sort", "index", "count"):
            assert result.task_runs[name].status == TaskRunStatus.PENDING

        # Edge count: qc→align, align→sort, sort→index, align→count = 4
        assert len(result.edges) == 4

        # Resources on align
        tr_align = result.task_runs["align"]
        assert len(tr_align.resource_profiles) == 2
        assert tr_align.resource_profiles[0]["cpus"] == 8
        assert tr_align.escalate_on == ["OOM", "TIMEOUT"]

        # Inputs on count
        tr_count = result.task_runs["count"]
        assert tr_count.inputs["bam"] == {"task": "align", "file": "bam"}
        assert tr_count.inputs["sample"] == {"parameter": "sample"}

        # Events: 1 RUN_CREATED + 5 TASK_CREATED
        events = get_events(session, run_id=result.run.id)
        assert len(events) == 6
