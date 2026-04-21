from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentbahn.projects.schemas import EventLogListResponse
from agentbahn.projects.schemas import ProjectListResponse
from agentbahn.projects.schemas import TaskListResponse


@dataclass(frozen=True)
class CommandResult:
    kind: Literal["message", "projects", "tasks", "events"]
    message: str | None = None
    projects: ProjectListResponse | None = None
    tasks: TaskListResponse | None = None
    events: EventLogListResponse | None = None


def message_result(message: str) -> CommandResult:
    return CommandResult(kind="message", message=message)
