"""
Otto model/io.py
----------------
Building-block models for workflow inputs, outputs, parameters, and resources.
These are pure Pydantic models — no database dependency.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Resource profiles
# ---------------------------------------------------------------------------

class ResourceProfile(BaseModel):
    """
    One resource tier for a task.
    Tasks can define multiple profiles; Otto escalates through them on failure.

    Fields:
        cpus:     number of CPU cores to request
        mem_gb:   memory in gigabytes
        walltime: maximum runtime in HH:MM:SS (or D-HH:MM:SS for Slurm)
        extra:    backend-specific overrides (e.g. {"partition": "gpu", "gpus": 1})
    """

    cpus:     int            = 1
    mem_gb:   float          = 4.0
    walltime: str            = "1:00:00"
    extra:    dict[str, Any] = Field(default_factory=dict)


class ResourceSpec(BaseModel):
    """
    Resource configuration for a task, combining escalation profiles and triggers.

    Fields:
        profiles:     ordered list of ResourceProfile; escalation moves forward
        escalate_on:  failure reasons that trigger escalation: OOM | TIMEOUT
                      (not USER — bugs in user code should not consume more resources)
    """

    profiles:    list[ResourceProfile] = Field(default_factory=list)
    escalate_on: list[str]             = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Workflow-level input declarations
# ---------------------------------------------------------------------------

class ParameterSpec(BaseModel):
    """
    Declaration of a scalar workflow-level input.

    Fields:
        type:        value type — plain or array variant (suffix []):
                     str | int | float | bool | str[] | int[] | float[] | bool[]
        default:     default value if the user does not supply one
        description: human-readable description shown in otto --help
        required:    if True, the user must supply a value; default must be None
    """

    type:        Literal[
                     "str", "int", "float", "bool",
                     "str[]", "int[]", "float[]", "bool[]",
                 ] = "str"
    default:     Any        = None
    description: str | None = None
    required:    bool       = False

    @property
    def is_array(self) -> bool:
        return self.type.endswith("[]")

    @property
    def base_type(self) -> str:
        return self.type.removesuffix("[]")

    @model_validator(mode="after")
    def check_required_has_no_default(self) -> ParameterSpec:
        if self.required and self.default is not None:
            raise ValueError("A required parameter cannot have a default value.")
        return self


class FileSpec(BaseModel):
    """
    Declaration of a file or directory workflow-level input.

    Fields:
        type:        file | file[] | directory
        extensions:  allowed extensions without leading dot (e.g. "bam", "fastq.gz");
                     empty means any extension is accepted
        required:    if True, the user must supply a value; default must be None
        default:     default path if the user does not supply one
        description: human-readable description shown in otto --help
    """

    type:        Literal["file", "file[]", "directory"] = "file"
    extensions:  list[str]  = Field(default_factory=list)
    required:    bool       = False
    default:     str | None = None
    description: str | None = None

    @property
    def is_array(self) -> bool:
        return self.type.endswith("[]")

    @property
    def base_type(self) -> str:
        return self.type.removesuffix("[]")

    @model_validator(mode="after")
    def check_required_has_no_default(self) -> FileSpec:
        if self.required and self.default is not None:
            raise ValueError("A required parameter cannot have a default value.")
        return self


# Smart union for WorkflowTemplate.inputs values.
# ParameterSpec handles str/int/float/bool types (+ [] variants),
# FileSpec handles file/directory types. Type value sets are disjoint.
ParamDecl = Annotated[ParameterSpec | FileSpec, Field(discriminator=None)]


# ---------------------------------------------------------------------------
# Input wiring — workflow sources
# ---------------------------------------------------------------------------

class ParameterInput(BaseModel):
    """Wire a task input from a workflow-level ParameterSpec."""

    parameter: str  # name in WorkflowTemplate.inputs


class FileInput(BaseModel):
    """Wire a task input from a workflow-level FileSpec (file or file[])."""

    file: str  # name in WorkflowTemplate.inputs


class FolderInput(BaseModel):
    """Wire a task input from a workflow-level FileSpec (directory)."""

    folder: str  # name in WorkflowTemplate.inputs


# ---------------------------------------------------------------------------
# Input wiring — task sources
# ---------------------------------------------------------------------------

class TaskParameterInput(BaseModel):
    """Wire a task input from a value output of an upstream task."""

    task:      str  # upstream task name
    parameter: str  # output name on that task — must be OutputSpec(type="value" or "value[]")


class TaskFileInput(BaseModel):
    """Wire a task input from a file output of an upstream task."""

    task: str  # upstream task name
    file: str  # output name on that task — must be OutputSpec(type="file" or "file[]")


class TaskFolderInput(BaseModel):
    """Wire a task input from a directory output of an upstream task."""

    task:   str  # upstream task name
    folder: str  # output name on that task — must be OutputSpec(type="directory")


# Union ordered so that two-field task types are tried before one-field workflow
# types — prevents {task: x, file: y} from being mis-parsed as FileInput(file=y).
# Within each group fields are disjoint, so no ambiguity remains.
InputSpec = Annotated[
    TaskFileInput | TaskParameterInput | TaskFolderInput
    | ParameterInput | FileInput | FolderInput,
    Field(discriminator=None),
]


# ---------------------------------------------------------------------------
# Output declarations
# ---------------------------------------------------------------------------

class OutputSpec(BaseModel):
    """
    Declaration of one output produced by a task.
    The actual URI is resolved at runtime by the engine — this is only the contract.

    Fields:
        type:    what the output is — plain or array variant (suffix []):
                 file | file[] | directory | value | value[]
        pattern: optional glob pattern the engine uses to discover the output file(s)
                 (e.g. "*.bam"). If None, the engine uses task-specific conventions.
    """

    type:    Literal["file", "file[]", "directory", "value", "value[]"] = "file"
    pattern: str | None = None

    @property
    def is_array(self) -> bool:
        return self.type.endswith("[]")

    @property
    def base_type(self) -> str:
        return self.type.removesuffix("[]")
