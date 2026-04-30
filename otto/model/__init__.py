"""
Otto model package.
Public API — import everything you need from here.
"""

from otto.model.io import (
    FileInput,
    FileSpec,
    FolderInput,
    InputSpec,
    OutputSpec,
    ParamDecl,
    ParameterInput,
    ParameterSpec,
    ResourceProfile,
    ResourceSpec,
    TaskFileInput,
    TaskFolderInput,
    TaskParameterInput,
)
from otto.model.workflow import TaskTemplate, WorkflowTemplate

__all__ = [
    # io — workflow-level declarations
    "FileSpec",
    "ParamDecl",
    "ParameterSpec",
    # io — input wiring (workflow sources)
    "FileInput",
    "FolderInput",
    "ParameterInput",
    # io — input wiring (task sources)
    "TaskFileInput",
    "TaskFolderInput",
    "TaskParameterInput",
    # io — other
    "InputSpec",
    "OutputSpec",
    "ResourceProfile",
    "ResourceSpec",
    # workflow
    "TaskTemplate",
    "WorkflowTemplate",
]
