from __future__ import annotations

from collections.abc import Callable

from textual.app import App
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Input
from textual.widgets import Static

from agentbahn.projects.schemas import ProjectListResponse
from agentbahn.projects.schemas import TaskListResponse
from agentbahn_tui.projects import fetch_projects
from agentbahn_tui.tasks import fetch_tasks


def get_placeholder_message() -> str:
    return "Enter /project list or /task list PROJECT_ID to fetch data from the HTTP API."


def format_projects_output(projects: ProjectListResponse) -> str:
    if not projects:
        return "No projects found."
    return "\n".join(
        f"{project.id}. {project.name} - {project.description}" for project in projects
    )


def format_tasks_output(tasks: TaskListResponse) -> str:
    if not tasks:
        return "No tasks found."
    return "\n".join(
        (
            f"{task.id}. [{task.status}] {task.title} "
            f"(feature: {task.feature_name}, assignee: {task.user_username})"
        )
        for task in tasks
    )


def run_tui_command(
    command: str,
    fetch_projects_command: Callable[[], ProjectListResponse],
    fetch_tasks_command: Callable[[int], TaskListResponse],
) -> str:
    normalized_command = command.strip()
    if normalized_command == "/project list":
        return format_projects_output(fetch_projects_command())

    command_parts = normalized_command.split()
    if command_parts[:2] != ["/task", "list"]:
        return f"Unknown command: {normalized_command or '<empty>'}"
    if len(command_parts) != 3:
        return "Usage: /task list PROJECT_ID"

    try:
        project_id = int(command_parts[2])
    except ValueError:
        return "PROJECT_ID must be an integer."
    return format_tasks_output(fetch_tasks_command(project_id))


class AgentbahnTui(App[None]):
    """Textual app with a simple slash-command interface."""

    TITLE = "Agentbahn TUI"

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
        with Vertical():
            yield Static(get_placeholder_message(), id="command-output")
            yield Input(placeholder="/project list", id="command-input")
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        output = self.query_one("#command-output", Static)
        output.update(
            run_tui_command(
                event.value,
                self._fetch_projects_command,
                self._fetch_tasks_command,
            )
        )
        event.input.value = ""


def run_tui() -> None:
    AgentbahnTui().run()
