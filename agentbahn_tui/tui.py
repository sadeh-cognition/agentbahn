from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from textual.app import App
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.containers import VerticalScroll
from textual.widgets import DataTable
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Input
from textual.widgets import Static

from agentbahn.projects.schemas import ProjectListResponse
from agentbahn.projects.schemas import TaskListResponse
from agentbahn_tui.projects import fetch_projects
from agentbahn_tui.tasks import fetch_tasks


def get_placeholder_message() -> str:
    return "Enter /project list or /task list PROJECT_ID to fetch data from projectbahn."


@dataclass(frozen=True)
class CommandResult:
    kind: Literal["message", "projects", "tasks"]
    message: str | None = None
    projects: ProjectListResponse | None = None
    tasks: TaskListResponse | None = None


def message_result(message: str) -> CommandResult:
    return CommandResult(kind="message", message=message)


def format_projects_output(projects: ProjectListResponse) -> CommandResult:
    if not projects:
        return message_result("No projects found.")
    return CommandResult(kind="projects", projects=projects)


def format_tasks_output(tasks: TaskListResponse) -> CommandResult:
    if not tasks:
        return message_result("No tasks found.")
    return CommandResult(kind="tasks", tasks=tasks)


def run_tui_command(
    command: str,
    fetch_projects_command: Callable[[], ProjectListResponse],
    fetch_tasks_command: Callable[[int], TaskListResponse],
) -> CommandResult:
    normalized_command = command.strip()
    if normalized_command == "/project list":
        return format_projects_output(fetch_projects_command())

    command_parts = normalized_command.split()
    if command_parts[:2] != ["/task", "list"]:
        return message_result(f"Unknown command: {normalized_command or '<empty>'}")
    if len(command_parts) != 3:
        return message_result("Usage: /task list PROJECT_ID")

    try:
        project_id = int(command_parts[2])
    except ValueError:
        return message_result("PROJECT_ID must be an integer.")
    return format_tasks_output(fetch_tasks_command(project_id))


class AgentbahnTui(App[None]):
    """Textual app with a simple slash-command interface."""

    TITLE = "Agentbahn TUI"
    CSS = """
    #app-body {
        height: 1fr;
    }

    #results-pane {
        height: 1fr;
        padding: 0 1;
    }

    #message-output {
        padding: 1 0;
    }

    #results-table {
        height: auto;
    }

    #command-input {
        dock: bottom;
        margin: 0 1 1 1;
    }
    """

    def __init__(
        self,
        *,
        fetch_projects_command: Callable[[], ProjectListResponse] = fetch_projects,
        fetch_tasks_command: Callable[[int], TaskListResponse] = fetch_tasks,
    ) -> None:
        super().__init__()
        self._fetch_projects_command = fetch_projects_command
        self._fetch_tasks_command = fetch_tasks_command

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="app-body"):
            with VerticalScroll(id="results-pane"):
                yield Static(get_placeholder_message(), id="message-output")
                yield DataTable(id="results-table")
            yield Input(placeholder="/project list", id="command-input")
        yield Footer()

    def on_mount(self) -> None:
        self._show_message(get_placeholder_message())

    def _show_message(self, message: str) -> None:
        message_output = self.query_one("#message-output", Static)
        results_table = self.query_one("#results-table", DataTable)
        message_output.display = True
        message_output.update(message)
        results_table.display = False
        results_table.clear(columns=True)

    def _show_projects(self, projects: ProjectListResponse) -> None:
        message_output = self.query_one("#message-output", Static)
        results_table = self.query_one("#results-table", DataTable)
        message_output.display = False
        results_table.display = True
        results_table.clear(columns=True)
        results_table.add_columns("ID", "Name", "Description")
        for project in projects:
            results_table.add_row(str(project.id), project.name, project.description)

    def _show_tasks(self, tasks: TaskListResponse) -> None:
        message_output = self.query_one("#message-output", Static)
        results_table = self.query_one("#results-table", DataTable)
        message_output.display = False
        results_table.display = True
        results_table.clear(columns=True)
        results_table.add_columns("ID", "Status", "Title", "Feature", "Assignee")
        for task in tasks:
            results_table.add_row(
                str(task.id),
                task.status,
                task.title,
                task.feature_name,
                task.user_username,
            )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        result = run_tui_command(
            event.value,
            self._fetch_projects_command,
            self._fetch_tasks_command,
        )
        if result.kind == "projects" and result.projects is not None:
            self._show_projects(result.projects)
        elif result.kind == "tasks" and result.tasks is not None:
            self._show_tasks(result.tasks)
        else:
            self._show_message(result.message or get_placeholder_message())
        event.input.value = ""


def run_tui() -> None:
    AgentbahnTui().run()
