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
from agentbahn_tui.projects import fetch_projects


def get_placeholder_message() -> str:
    return "Enter /project list to fetch projects from the HTTP API."


def format_projects_output(projects: ProjectListResponse) -> str:
    if not projects:
        return "No projects found."
    return "\n".join(
        f"{project.id}. {project.name} - {project.description}" for project in projects
    )


def run_tui_command(
    command: str,
    fetch_projects_command: Callable[[], ProjectListResponse],
) -> str:
    normalized_command = command.strip()
    if normalized_command != "/project list":
        return f"Unknown command: {normalized_command or '<empty>'}"
    return format_projects_output(fetch_projects_command())


class AgentbahnTui(App[None]):
    """Textual app with a simple slash-command interface."""

    TITLE = "Agentbahn TUI"

    def __init__(
        self,
        *,
        fetch_projects_command: Callable[[], ProjectListResponse] = fetch_projects,
    ) -> None:
        super().__init__()
        self._fetch_projects_command = fetch_projects_command

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static(get_placeholder_message(), id="command-output")
            yield Input(placeholder="/project list", id="command-input")
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        output = self.query_one("#command-output", Static)
        output.update(run_tui_command(event.value, self._fetch_projects_command))
        event.input.value = ""


def run_tui() -> None:
    AgentbahnTui().run()
