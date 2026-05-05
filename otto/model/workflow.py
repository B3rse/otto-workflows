"""
Otto model/workflow.py
----------------------
Top-level workflow and task template models.
These are pure Pydantic models — no database dependency.

A WorkflowTemplate is the authored definition of a workflow. The planner
reads it and creates the corresponding Run + TaskRun + Edge records in the DB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import networkx as nx
import yaml
from pydantic import BaseModel, Field, model_validator

from otto.model.io import (
    FileInput,
    FileSpec,
    FolderInput,
    InputSpec,
    OutputSpec,
    ParamDecl,
    ParameterInput,
    ParameterSpec,
    ResourceSpec,
    TaskFileInput,
    TaskFolderInput,
    TaskParameterInput,
)

# ---------------------------------------------------------------------------
# Task template
# ---------------------------------------------------------------------------

class TaskTemplate(BaseModel):
    """
    Definition of one task within a workflow.

    Fields:
        name:           unique identifier within the workflow (used in dependencies
                        and input wiring)
        engine:         what runs the command: shell | nextflow | cwl | snakemake | ...
        backend:        where it runs: local | slurm | aws_batch | ...
        cmd:            command string; may reference inputs/outputs via {inputs.x}
                        and {outputs.y} placeholders resolved at runtime
        dependencies:   list of task names that must succeed before this task runs
        inputs:         named input slots and how they are wired
        outputs:        named output slots and what they produce
        resources:      resource profiles and escalation config
        scatter:        if set, fan out this task over each element of this input name
                        (the named input must resolve to an array artifact at runtime)
        scatter_method: how to fan out when scatter is set:
                        - flat:    one shard per element across all upstream shards (N×K)
                        - grouped: one shard per upstream shard, each receiving its K
                                   elements as an array (preserves shard grouping)
    """

    name:           str
    engine:         str = "shell"
    backend:        str = "local"
    cmd:            str | None = None
    dependencies:   list[str]             = Field(default_factory=list)
    inputs:         dict[str, InputSpec]  = Field(default_factory=dict)
    outputs:        dict[str, OutputSpec] = Field(default_factory=dict)
    resources:      ResourceSpec          = Field(default_factory=ResourceSpec)
    scatter:        str | None = None
    scatter_method: Literal["flat", "grouped"] = "flat"

    @model_validator(mode="after")
    def check_scatter_method(self) -> TaskTemplate:
        if self.scatter_method == "grouped" and self.scatter is None:
            raise ValueError("scatter_method='grouped' requires scatter to be set.")
        return self


# ---------------------------------------------------------------------------
# Workflow template
# ---------------------------------------------------------------------------

class WorkflowTemplate(BaseModel):
    """
    The complete authored definition of a workflow.

    Fields:
        name:        workflow identifier (used as Run.workflow_name in the DB)
        version:     optional version string
        description: optional human-readable description
        inputs:      named workflow-level inputs the user supplies at run time
                     (ParameterSpec for scalars, FileSpec for files/directories)
        tasks:       ordered list of TaskTemplates (order does not imply execution
                     order — that is determined by dependencies)

    Validation (runs automatically on construction):
        - No duplicate task names
        - All dependency references point to existing tasks
        - All input wiring references point to existing workflow inputs or task outputs
        - Wiring types are consistent with declared input/output types
        - If a task wires from another task's output, that task must appear in dependencies
        - The dependency graph is a DAG (no cycles)
        - If scatter is set, the named slot must exist in the task's declared inputs
    """

    name:        str
    version:     str | None = None
    description: str | None = None
    inputs:      dict[str, ParamDecl]  = Field(default_factory=dict)
    tasks:       list[TaskTemplate]    = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_workflow(self) -> WorkflowTemplate:
        task_map = {t.name: t for t in self.tasks}

        self._check_no_duplicate_names(task_map)
        self._check_dependencies(task_map)
        self._check_input_wiring(task_map)
        self._check_no_cycles(task_map)

        return self

    def _check_no_duplicate_names(self, task_map: dict[str, TaskTemplate]) -> None:
        if len(task_map) != len(self.tasks):
            names = [t.name for t in self.tasks]
            dupes = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Duplicate task names: {sorted(dupes)}")

    def _check_dependencies(self, task_map: dict[str, TaskTemplate]) -> None:
        for task in self.tasks:
            for dep in task.dependencies:
                if dep not in task_map:
                    raise ValueError(
                        f"Task '{task.name}' depends on unknown task '{dep}'."
                    )

    def _check_input_wiring(self, task_map: dict[str, TaskTemplate]) -> None:
        for task in self.tasks:
            if task.scatter and task.scatter not in task.inputs:
                raise ValueError(
                    f"Task '{task.name}' scatter '{task.scatter}' does not reference "
                    f"a declared input slot."
                )
            for slot, spec in task.inputs.items():

                # -- workflow-level sources --

                if isinstance(spec, ParameterInput):
                    if spec.parameter not in self.inputs:
                        raise ValueError(
                            f"Task '{task.name}' input '{slot}' references "
                            f"unknown parameter '{spec.parameter}'."
                        )
                    decl = self.inputs[spec.parameter]
                    if not isinstance(decl, ParameterSpec):
                        raise ValueError(
                            f"Task '{task.name}' input '{slot}': '{spec.parameter}' "
                            f"is a file/folder input; use {{file:}} or {{folder:}} instead."
                        )

                elif isinstance(spec, FileInput):
                    if spec.file not in self.inputs:
                        raise ValueError(
                            f"Task '{task.name}' input '{slot}' references "
                            f"unknown file input '{spec.file}'."
                        )
                    decl = self.inputs[spec.file]
                    if not isinstance(decl, FileSpec):
                        raise ValueError(
                            f"Task '{task.name}' input '{slot}': '{spec.file}' "
                            f"is a scalar parameter; use {{parameter:}} instead."
                        )
                    if decl.base_type != "file":
                        raise ValueError(
                            f"Task '{task.name}' input '{slot}': '{spec.file}' "
                            f"is a folder input; use {{folder:}} instead."
                        )

                elif isinstance(spec, FolderInput):
                    if spec.folder not in self.inputs:
                        raise ValueError(
                            f"Task '{task.name}' input '{slot}' references "
                            f"unknown folder input '{spec.folder}'."
                        )
                    decl = self.inputs[spec.folder]
                    if not isinstance(decl, FileSpec):
                        raise ValueError(
                            f"Task '{task.name}' input '{slot}': '{spec.folder}' "
                            f"is a scalar parameter; use {{parameter:}} instead."
                        )
                    if decl.type != "directory":
                        raise ValueError(
                            f"Task '{task.name}' input '{slot}': '{spec.folder}' "
                            f"is a file input; use {{file:}} instead."
                        )

                # -- task-level sources --

                elif isinstance(spec, TaskFileInput):
                    out = self._resolve_task_output(
                        task, slot, spec.task, spec.file, task_map
                    )
                    if out.base_type != "file":
                        raise ValueError(
                            f"Task '{task.name}' input '{slot}': output '{spec.file}' "
                            f"on task '{spec.task}' is not a file output; "
                            f"use {{parameter:}} or {{folder:}} instead."
                        )

                elif isinstance(spec, TaskParameterInput):
                    out = self._resolve_task_output(
                        task, slot, spec.task, spec.parameter, task_map
                    )
                    if out.base_type != "value":
                        raise ValueError(
                            f"Task '{task.name}' input '{slot}': output '{spec.parameter}' "
                            f"on task '{spec.task}' is not a value output; "
                            f"use {{file:}} or {{folder:}} instead."
                        )

                elif isinstance(spec, TaskFolderInput):
                    out = self._resolve_task_output(
                        task, slot, spec.task, spec.folder, task_map
                    )
                    if out.type != "directory":
                        raise ValueError(
                            f"Task '{task.name}' input '{slot}': output '{spec.folder}' "
                            f"on task '{spec.task}' is not a directory output; "
                            f"use {{file:}} or {{parameter:}} instead."
                        )

    def _resolve_task_output(
        self,
        task: TaskTemplate,
        slot: str,
        upstream_name: str,
        output_name: str,
        task_map: dict[str, TaskTemplate],
    ) -> OutputSpec:
        """Resolve and return an upstream task output, raising ValueError if missing."""
        if upstream_name not in task_map:
            raise ValueError(
                f"Task '{task.name}' input '{slot}' references "
                f"unknown task '{upstream_name}'."
            )
        if upstream_name not in task.dependencies:
            raise ValueError(
                f"Task '{task.name}' input '{slot}' wires from task '{upstream_name}' "
                f"but '{upstream_name}' is not listed in '{task.name}'.dependencies."
            )
        upstream = task_map[upstream_name]
        if output_name not in upstream.outputs:
            raise ValueError(
                f"Task '{task.name}' input '{slot}' references "
                f"output '{output_name}' which does not exist on task '{upstream_name}'."
            )
        return upstream.outputs[output_name]

    def _check_no_cycles(self, task_map: dict[str, TaskTemplate]) -> None:
        graph: nx.DiGraph = nx.DiGraph()
        for task in self.tasks:
            graph.add_node(task.name)
            for dep in task.dependencies:
                graph.add_edge(dep, task.name)
        if not nx.is_directed_acyclic_graph(graph):
            cycles = list(nx.simple_cycles(graph))
            raise ValueError(f"Workflow dependency graph contains cycles: {cycles}")

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> WorkflowTemplate:
        """Load and validate a WorkflowTemplate from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Serialize this template to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(exclude_none=True), f, sort_keys=False)

    def task(self, name: str) -> TaskTemplate:
        """Look up a task by name. Raises KeyError if not found."""
        for t in self.tasks:
            if t.name == name:
                return t
        raise KeyError(f"No task named '{name}' in workflow '{self.name}'.")

    def root_tasks(self) -> list[TaskTemplate]:
        """Return tasks with no dependencies — the starting points of the DAG."""
        return [t for t in self.tasks if not t.dependencies]

    def model_dump_for_run(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for storing in Run.parameters."""
        return self.model_dump(exclude_none=True)
