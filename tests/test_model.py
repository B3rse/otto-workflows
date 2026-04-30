"""
tests/test_model.py
-------------------
Tests for the model layer — WorkflowTemplate, TaskTemplate, and all IO models.
No database involved; pure Pydantic validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_workflow(**kwargs) -> WorkflowTemplate:
    """Minimal valid workflow with overridable fields."""
    defaults = dict(name="test_wf", tasks=[])
    return WorkflowTemplate(**(defaults | kwargs))


def make_task(**kwargs) -> TaskTemplate:
    """Minimal valid task with overridable fields."""
    defaults = dict(name="step_a", engine="shell", backend="local")
    return TaskTemplate(**(defaults | kwargs))


# ---------------------------------------------------------------------------
# ResourceProfile
# ---------------------------------------------------------------------------

class TestResourceProfile:
    def test_defaults(self):
        p = ResourceProfile()
        assert p.cpus == 1
        assert p.mem_gb == 4.0
        assert p.walltime == "1:00:00"
        assert p.extra == {}

    def test_custom_values(self):
        p = ResourceProfile(cpus=8, mem_gb=32.0, walltime="4:00:00")
        assert p.cpus == 8
        assert p.mem_gb == 32.0

    def test_extra_backend_fields(self):
        p = ResourceProfile(extra={"partition": "gpu", "gpus": 1})
        assert p.extra["partition"] == "gpu"


# ---------------------------------------------------------------------------
# ResourceSpec
# ---------------------------------------------------------------------------

class TestResourceSpec:
    def test_defaults(self):
        rs = ResourceSpec()
        assert rs.profiles == []
        assert rs.escalate_on == []

    def test_with_profiles(self):
        rs = ResourceSpec(
            profiles=[
                ResourceProfile(cpus=4, mem_gb=16, walltime="2:00:00"),
                ResourceProfile(cpus=8, mem_gb=32, walltime="4:00:00"),
            ],
            escalate_on=["OOM", "TIMEOUT"],
        )
        assert len(rs.profiles) == 2
        assert rs.escalate_on == ["OOM", "TIMEOUT"]


# ---------------------------------------------------------------------------
# ParameterSpec
# ---------------------------------------------------------------------------

class TestParameterSpec:
    def test_defaults(self):
        p = ParameterSpec()
        assert p.type == "str"
        assert p.default is None
        assert p.required is False

    def test_required_no_default(self):
        p = ParameterSpec(type="str", required=True)
        assert p.required is True
        assert p.default is None

    def test_optional_with_default(self):
        p = ParameterSpec(type="str", default="hg38")
        assert p.default == "hg38"

    def test_required_with_default_raises(self):
        with pytest.raises(ValueError, match="required parameter cannot have a default"):
            ParameterSpec(required=True, default="something")

    def test_scalar_is_not_array(self):
        p = ParameterSpec(type="int")
        assert p.is_array is False
        assert p.base_type == "int"

    def test_array_type(self):
        p = ParameterSpec(type="str[]")
        assert p.type == "str[]"
        assert p.is_array is True
        assert p.base_type == "str"

    def test_all_scalar_types(self):
        for t in ("str", "int", "float", "bool"):
            assert ParameterSpec(type=t).type == t

    def test_all_array_types(self):
        for t in ("str[]", "int[]", "float[]", "bool[]"):
            assert ParameterSpec(type=t).is_array is True

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            ParameterSpec(type="blob")

    def test_path_type_no_longer_valid(self):
        with pytest.raises(ValueError):
            ParameterSpec(type="path")


# ---------------------------------------------------------------------------
# FileSpec
# ---------------------------------------------------------------------------

class TestFileSpec:
    def test_defaults(self):
        f = FileSpec()
        assert f.type == "file"
        assert f.extensions == []
        assert f.required is False
        assert f.default is None
        assert f.is_array is False
        assert f.base_type == "file"

    def test_file_array(self):
        f = FileSpec(type="file[]")
        assert f.is_array is True
        assert f.base_type == "file"

    def test_directory(self):
        f = FileSpec(type="directory")
        assert f.type == "directory"
        assert f.is_array is False

    def test_extensions_without_dot(self):
        f = FileSpec(extensions=["bam", "cram", "fastq.gz"])
        assert f.extensions == ["bam", "cram", "fastq.gz"]

    def test_required_no_default(self):
        f = FileSpec(required=True)
        assert f.required is True
        assert f.default is None

    def test_required_with_default_raises(self):
        with pytest.raises(ValueError, match="required parameter cannot have a default"):
            FileSpec(required=True, default="/path/to/file.bam")

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            FileSpec(type="blob")


# ---------------------------------------------------------------------------
# Input wiring — construction
# ---------------------------------------------------------------------------

class TestInputWiring:
    def test_parameter_input(self):
        spec = ParameterInput(parameter="genome")
        assert spec.parameter == "genome"

    def test_file_input(self):
        spec = FileInput(file="sample_bam")
        assert spec.file == "sample_bam"

    def test_folder_input(self):
        spec = FolderInput(folder="reference_dir")
        assert spec.folder == "reference_dir"

    def test_task_file_input(self):
        spec = TaskFileInput(task="align", file="bam")
        assert spec.task == "align"
        assert spec.file == "bam"

    def test_task_parameter_input(self):
        spec = TaskParameterInput(task="count", parameter="n")
        assert spec.task == "count"
        assert spec.parameter == "n"

    def test_task_folder_input(self):
        spec = TaskFolderInput(task="setup", folder="refs")
        assert spec.task == "setup"
        assert spec.folder == "refs"


# ---------------------------------------------------------------------------
# OutputSpec
# ---------------------------------------------------------------------------

class TestOutputSpec:
    def test_default_type_is_file(self):
        o = OutputSpec()
        assert o.type == "file"
        assert o.is_array is False
        assert o.base_type == "file"

    def test_file_array(self):
        o = OutputSpec(type="file[]")
        assert o.is_array is True
        assert o.base_type == "file"

    def test_directory_type(self):
        o = OutputSpec(type="directory")
        assert o.type == "directory"
        assert o.is_array is False

    def test_value_type(self):
        o = OutputSpec(type="value")
        assert o.type == "value"
        assert o.is_array is False

    def test_value_array(self):
        o = OutputSpec(type="value[]")
        assert o.is_array is True
        assert o.base_type == "value"

    def test_with_pattern(self):
        o = OutputSpec(type="file", pattern="*.bam")
        assert o.pattern == "*.bam"

    def test_array_type_no_longer_valid(self):
        with pytest.raises(ValueError):
            OutputSpec(type="array")

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            OutputSpec(type="blob")


# ---------------------------------------------------------------------------
# TaskTemplate
# ---------------------------------------------------------------------------

class TestTaskTemplate:
    def test_minimal(self):
        t = make_task()
        assert t.name == "step_a"
        assert t.engine == "shell"
        assert t.backend == "local"
        assert t.dependencies == []
        assert t.inputs == {}
        assert t.outputs == {}

    def test_with_resources(self):
        t = make_task(
            resources=ResourceSpec(
                profiles=[ResourceProfile(cpus=4, mem_gb=16, walltime="2:00:00")],
                escalate_on=["OOM"],
            )
        )
        assert t.resources.profiles[0].cpus == 4
        assert t.resources.escalate_on == ["OOM"]

    def test_with_scatter_defaults_to_flat(self):
        t = make_task(scatter="bam_files")
        assert t.scatter == "bam_files"
        assert t.scatter_method == "flat"

    def test_scatter_method_grouped(self):
        t = make_task(scatter="bam_files", scatter_method="grouped")
        assert t.scatter_method == "grouped"

    def test_scatter_method_grouped_without_scatter_raises(self):
        with pytest.raises(ValueError, match="scatter_method='grouped' requires scatter"):
            make_task(scatter_method="grouped")

    def test_scatter_method_invalid_raises(self):
        with pytest.raises(ValueError):
            make_task(scatter="bam_files", scatter_method="cross")


# ---------------------------------------------------------------------------
# WorkflowTemplate — construction
# ---------------------------------------------------------------------------

class TestWorkflowTemplate:
    def test_minimal(self):
        wf = make_workflow()
        assert wf.name == "test_wf"
        assert wf.tasks == []
        assert wf.inputs == {}

    def test_with_tasks(self):
        wf = make_workflow(tasks=[make_task(name="a"), make_task(name="b")])
        assert len(wf.tasks) == 2

    def test_with_parameter_input(self):
        wf = make_workflow(inputs={"sample": ParameterSpec(required=True)})
        assert "sample" in wf.inputs
        assert wf.inputs["sample"].required is True  # type: ignore[union-attr]

    def test_with_file_input(self):
        wf = make_workflow(inputs={"bam": FileSpec(type="file", extensions=["bam", "cram"])})
        assert wf.inputs["bam"].type == "file"  # type: ignore[union-attr]

    def test_with_file_array_input(self):
        wf = make_workflow(inputs={"samples": FileSpec(type="file[]", required=True)})
        assert wf.inputs["samples"].is_array is True  # type: ignore[union-attr]

    def test_with_folder_input(self):
        wf = make_workflow(inputs={"ref": FileSpec(type="directory")})
        assert wf.inputs["ref"].type == "directory"  # type: ignore[union-attr]

    def test_root_tasks(self):
        wf = make_workflow(tasks=[
            make_task(name="a"),
            make_task(name="b", dependencies=["a"]),
        ])
        roots = wf.root_tasks()
        assert len(roots) == 1
        assert roots[0].name == "a"

    def test_task_lookup(self):
        wf = make_workflow(tasks=[make_task(name="align")])
        assert wf.task("align").name == "align"

    def test_task_lookup_missing_raises(self):
        wf = make_workflow()
        with pytest.raises(KeyError):
            wf.task("nonexistent")

    def test_model_dump_for_run(self):
        wf = make_workflow(
            name="dump_test",
            inputs={"sample": ParameterSpec(type="str", required=True)},
            tasks=[make_task(name="step")],
        )
        d = wf.model_dump_for_run()
        assert isinstance(d, dict)
        assert d["name"] == "dump_test"
        assert "inputs" in d
        assert "tasks" in d


# ---------------------------------------------------------------------------
# WorkflowTemplate — validation
# ---------------------------------------------------------------------------

class TestWorkflowValidation:

    # --- DAG structure ---

    def test_duplicate_task_names_raises(self):
        with pytest.raises(ValueError, match="Duplicate task names"):
            make_workflow(tasks=[make_task(name="a"), make_task(name="a")])

    def test_unknown_dependency_raises(self):
        with pytest.raises(ValueError, match="unknown task 'ghost'"):
            make_workflow(tasks=[make_task(name="a", dependencies=["ghost"])])

    def test_cycle_raises(self):
        with pytest.raises(ValueError, match="cycles"):
            make_workflow(tasks=[
                make_task(name="a", dependencies=["b"]),
                make_task(name="b", dependencies=["a"]),
            ])

    def test_self_dependency_cycle_raises(self):
        with pytest.raises(ValueError, match="cycles"):
            make_workflow(tasks=[make_task(name="a", dependencies=["a"])])

    # --- workflow-level input wiring errors ---

    def test_unknown_parameter_input_raises(self):
        with pytest.raises(ValueError, match="unknown parameter 'genome'"):
            make_workflow(
                inputs={},
                tasks=[make_task(name="a", inputs={"ref": ParameterInput(parameter="genome")})],
            )

    def test_parameter_input_on_file_spec_raises(self):
        with pytest.raises(ValueError, match="use {file:} or {folder:} instead"):
            make_workflow(
                inputs={"reads": FileSpec(type="file[]")},
                tasks=[make_task(name="a", inputs={"r": ParameterInput(parameter="reads")})],
            )

    def test_unknown_file_input_raises(self):
        with pytest.raises(ValueError, match="unknown file input 'missing'"):
            make_workflow(
                inputs={},
                tasks=[make_task(name="a", inputs={"f": FileInput(file="missing")})],
            )

    def test_file_input_on_parameter_spec_raises(self):
        with pytest.raises(ValueError, match="use {parameter:} instead"):
            make_workflow(
                inputs={"genome": ParameterSpec(type="str")},
                tasks=[make_task(name="a", inputs={"g": FileInput(file="genome")})],
            )

    def test_file_input_on_folder_spec_raises(self):
        with pytest.raises(ValueError, match="use {folder:} instead"):
            make_workflow(
                inputs={"ref": FileSpec(type="directory")},
                tasks=[make_task(name="a", inputs={"r": FileInput(file="ref")})],
            )

    def test_unknown_folder_input_raises(self):
        with pytest.raises(ValueError, match="unknown folder input 'missing'"):
            make_workflow(
                inputs={},
                tasks=[make_task(name="a", inputs={"d": FolderInput(folder="missing")})],
            )

    def test_folder_input_on_file_spec_raises(self):
        with pytest.raises(ValueError, match="use {file:} instead"):
            make_workflow(
                inputs={"reads": FileSpec(type="file[]")},
                tasks=[make_task(name="a", inputs={"r": FolderInput(folder="reads")})],
            )

    def test_folder_input_on_parameter_spec_raises(self):
        with pytest.raises(ValueError, match="use {parameter:} instead"):
            make_workflow(
                inputs={"genome": ParameterSpec(type="str")},
                tasks=[make_task(name="a", inputs={"g": FolderInput(folder="genome")})],
            )

    # --- scatter ---

    def test_scatter_slot_not_in_task_inputs_raises(self):
        with pytest.raises(ValueError, match="scatter 'bam_files' does not reference"):
            make_workflow(
                tasks=[
                    make_task(
                        name="a",
                        scatter="bam_files",
                        # bam_files not declared in inputs
                    )
                ]
            )

    def test_valid_scatter_in_workflow(self):
        wf = make_workflow(
            inputs={"reads": FileSpec(type="file[]", required=True)},
            tasks=[
                make_task(
                    name="process",
                    scatter="fq",
                    inputs={"fq": FileInput(file="reads")},
                    outputs={"result": OutputSpec(type="file")},
                )
            ],
        )
        assert wf.task("process").scatter == "fq"

    # --- task-level output wiring errors ---

    def test_task_file_input_without_dependency_raises(self):
        with pytest.raises(ValueError, match="not listed in 'b'.dependencies"):
            make_workflow(
                tasks=[
                    make_task(name="a", outputs={"bam": OutputSpec(type="file")}),
                    make_task(name="b", inputs={"x": TaskFileInput(task="a", file="bam")}),
                ]
            )

    def test_task_parameter_input_without_dependency_raises(self):
        with pytest.raises(ValueError, match="not listed in 'b'.dependencies"):
            make_workflow(
                tasks=[
                    make_task(name="a", outputs={"n": OutputSpec(type="value")}),
                    make_task(name="b", inputs={"x": TaskParameterInput(task="a", parameter="n")}),
                ]
            )

    def test_task_folder_input_without_dependency_raises(self):
        with pytest.raises(ValueError, match="not listed in 'b'.dependencies"):
            make_workflow(
                tasks=[
                    make_task(name="a", outputs={"refs": OutputSpec(type="directory")}),
                    make_task(name="b", inputs={"r": TaskFolderInput(task="a", folder="refs")}),
                ]
            )

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="unknown task 'missing'"):
            make_workflow(
                tasks=[
                    make_task(name="b", inputs={"x": TaskFileInput(task="missing", file="out")}),
                ]
            )

    def test_unknown_task_output_raises(self):
        with pytest.raises(ValueError, match="output 'bam' which does not exist"):
            make_workflow(
                tasks=[
                    make_task(name="a", outputs={"result": OutputSpec()}),
                    make_task(
                        name="b",
                        dependencies=["a"],
                        inputs={"x": TaskFileInput(task="a", file="bam")},
                    ),
                ]
            )

    def test_task_output_type_mismatch_raises(self):
        with pytest.raises(ValueError, match="is not a file output"):
            make_workflow(
                tasks=[
                    make_task(name="a", outputs={"n": OutputSpec(type="value")}),
                    make_task(
                        name="b",
                        dependencies=["a"],
                        inputs={"x": TaskFileInput(task="a", file="n")},
                    ),
                ]
            )

    def test_task_value_output_type_mismatch_raises(self):
        with pytest.raises(ValueError, match="is not a value output"):
            make_workflow(
                tasks=[
                    make_task(name="a", outputs={"bam": OutputSpec(type="file")}),
                    make_task(
                        name="b",
                        dependencies=["a"],
                        inputs={"x": TaskParameterInput(task="a", parameter="bam")},
                    ),
                ]
            )

    # --- valid wiring ---

    def test_valid_parameter_input_wiring(self):
        wf = make_workflow(
            inputs={"genome": ParameterSpec(default="hg38")},
            tasks=[make_task(name="a", inputs={"ref": ParameterInput(parameter="genome")})],
        )
        assert wf.task("a").inputs["ref"].parameter == "genome"  # type: ignore[union-attr]

    def test_valid_file_input_wiring(self):
        wf = make_workflow(
            inputs={"sample_bam": FileSpec(type="file", extensions=["bam"])},
            tasks=[make_task(name="a", inputs={"bam": FileInput(file="sample_bam")})],
        )
        assert wf.task("a").inputs["bam"].file == "sample_bam"  # type: ignore[union-attr]

    def test_valid_file_array_input_wiring(self):
        wf = make_workflow(
            inputs={"reads": FileSpec(type="file[]", extensions=["fastq.gz"])},
            tasks=[make_task(name="a", inputs={"fq": FileInput(file="reads")})],
        )
        assert wf.inputs["reads"].is_array is True  # type: ignore[union-attr]

    def test_valid_folder_input_wiring(self):
        wf = make_workflow(
            inputs={"ref_dir": FileSpec(type="directory")},
            tasks=[make_task(name="a", inputs={"ref": FolderInput(folder="ref_dir")})],
        )
        assert wf.task("a").inputs["ref"].folder == "ref_dir"  # type: ignore[union-attr]

    def test_valid_task_file_input_wiring(self):
        wf = make_workflow(
            tasks=[
                make_task(name="a", outputs={"bam": OutputSpec(type="file")}),
                make_task(
                    name="b",
                    dependencies=["a"],
                    inputs={"x": TaskFileInput(task="a", file="bam")},
                ),
            ]
        )
        assert wf.task("b").inputs["x"].file == "bam"  # type: ignore[union-attr]

    def test_valid_task_file_input_on_array_output(self):
        wf = make_workflow(
            tasks=[
                make_task(name="a", outputs={"bams": OutputSpec(type="file[]")}),
                make_task(
                    name="b",
                    dependencies=["a"],
                    inputs={"x": TaskFileInput(task="a", file="bams")},
                ),
            ]
        )
        assert wf.task("b").inputs["x"].task == "a"  # type: ignore[union-attr]

    def test_valid_task_parameter_input_wiring(self):
        wf = make_workflow(
            tasks=[
                make_task(name="a", outputs={"count": OutputSpec(type="value")}),
                make_task(
                    name="b",
                    dependencies=["a"],
                    inputs={"n": TaskParameterInput(task="a", parameter="count")},
                ),
            ]
        )
        assert wf.task("b").inputs["n"].parameter == "count"  # type: ignore[union-attr]

    def test_valid_task_folder_input_wiring(self):
        wf = make_workflow(
            tasks=[
                make_task(name="a", outputs={"refs": OutputSpec(type="directory")}),
                make_task(
                    name="b",
                    dependencies=["a"],
                    inputs={"ref": TaskFolderInput(task="a", folder="refs")},
                ),
            ]
        )
        assert wf.task("b").inputs["ref"].folder == "refs"  # type: ignore[union-attr]

    def test_valid_task_parameter_array_input_wiring(self):
        wf = make_workflow(
            tasks=[
                make_task(name="a", outputs={"counts": OutputSpec(type="value[]")}),
                make_task(
                    name="b",
                    dependencies=["a"],
                    inputs={"ns": TaskParameterInput(task="a", parameter="counts")},
                ),
            ]
        )
        assert wf.task("a").outputs["counts"].is_array is True


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------

class TestYamlRoundTrip:
    def test_from_yaml(self, tmp_path: Path):
        data = {
            "name": "rna_seq",
            "version": "1.0",
            "inputs": {
                "sample": {"type": "str", "required": True},
                "genome": {"type": "str", "default": "hg38"},
                "fastq":  {"type": "file[]", "extensions": ["fastq.gz"], "required": True},
                "ref_dir": {"type": "directory"},
            },
            "tasks": [
                {
                    "name": "align",
                    "engine": "shell",
                    "backend": "slurm",
                    "cmd": "bwa mem {inputs.genome} {inputs.fastq} > {outputs.bam}",
                    "inputs": {
                        "fastq":  {"file": "fastq"},
                        "genome": {"parameter": "genome"},
                        "ref":    {"folder": "ref_dir"},
                    },
                    "outputs": {
                        "bam": {"type": "file"},
                        "log": {"type": "value"},
                    },
                    "resources": {
                        "profiles": [
                            {"cpus": 8, "mem_gb": 32.0, "walltime": "4:00:00"}
                        ],
                        "escalate_on": ["OOM", "TIMEOUT"],
                    },
                },
                {
                    "name": "sort",
                    "engine": "shell",
                    "backend": "slurm",
                    "cmd": "samtools sort {inputs.bam} -o {outputs.sorted_bam}",
                    "dependencies": ["align"],
                    "inputs": {
                        "bam": {"task": "align", "file": "bam"},
                        "run_id": {"task": "align", "parameter": "log"},
                    },
                    "outputs": {"sorted_bam": {"type": "file"}},
                },
            ],
        }
        yaml_file = tmp_path / "wf.yaml"
        yaml_file.write_text(yaml.dump(data))

        wf = WorkflowTemplate.from_yaml(yaml_file)

        assert wf.name == "rna_seq"
        assert wf.version == "1.0"
        assert wf.inputs["sample"].required is True       # type: ignore[union-attr]
        assert wf.inputs["genome"].default == "hg38"      # type: ignore[union-attr]
        assert wf.inputs["fastq"].is_array is True        # type: ignore[union-attr]
        assert wf.inputs["fastq"].extensions == ["fastq.gz"]  # type: ignore[union-attr]
        assert wf.inputs["ref_dir"].type == "directory"   # type: ignore[union-attr]
        assert len(wf.tasks) == 2
        assert wf.task("align").resources.profiles[0].cpus == 8
        assert wf.task("sort").dependencies == ["align"]
        assert wf.task("align").inputs["fastq"].file == "fastq"       # type: ignore[union-attr]
        assert wf.task("align").inputs["genome"].parameter == "genome" # type: ignore[union-attr]
        assert wf.task("align").inputs["ref"].folder == "ref_dir"      # type: ignore[union-attr]
        assert wf.task("sort").inputs["bam"].file == "bam"             # type: ignore[union-attr]
        assert wf.task("sort").inputs["run_id"].parameter == "log"     # type: ignore[union-attr]

    def test_to_yaml_and_back(self, tmp_path: Path):
        wf = make_workflow(
            name="roundtrip",
            inputs={
                "sample": ParameterSpec(type="str", required=True),
                "reads":  FileSpec(type="file[]", extensions=["fastq.gz"], required=True),
                "ref":    FileSpec(type="directory"),
            },
            tasks=[
                make_task(
                    name="step",
                    inputs={
                        "s":   ParameterInput(parameter="sample"),
                        "fq":  FileInput(file="reads"),
                        "ref": FolderInput(folder="ref"),
                    },
                    outputs={"result": OutputSpec(type="file")},
                )
            ],
        )
        path = tmp_path / "out.yaml"
        wf.to_yaml(path)
        wf2 = WorkflowTemplate.from_yaml(path)

        assert wf2.name == wf.name
        assert wf2.tasks[0].name == wf.tasks[0].name
        assert wf2.inputs["reads"].is_array is True          # type: ignore[union-attr]
        assert wf2.inputs["reads"].extensions == ["fastq.gz"]  # type: ignore[union-attr]
        assert wf2.inputs["ref"].type == "directory"         # type: ignore[union-attr]
        assert wf2.task("step").inputs["fq"].file == "reads"   # type: ignore[union-attr]
        assert wf2.task("step").inputs["ref"].folder == "ref"  # type: ignore[union-attr]
