from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from textual.binding import Binding
from textual.app import App
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.containers import VerticalScroll
from textual.widgets import DataTable
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Input
from textual.widgets import Static

from agentbahn.projects.schemas import EventLogListResponse
from agentbahn.projects.schemas import ProjectListResponse
from agentbahn.projects.schemas import TaskListResponse
from agentbahn_tui.project_events import fetch_project_events
from agentbahn_tui.projects import fetch_projects
from agentbahn_tui.tasks import fetch_tasks


def get_placeholder_message() -> str:
    return (
        "Enter /project list, /project event list PROJECT_ID, or /task list PROJECT_ID "
        "to fetch data from projectbahn."
    )


@dataclass
class CommandHistory:
    commands: list[str]
    draft: str = ""
    index: int | None = None

    def record(self, command: str) -> None:
        normalized_command = command.strip()
        if not normalized_command:
            self.reset_navigation()
            return
        self.commands.append(normalized_command)
        self.reset_navigation()

    def previous(self, current_value: str) -> str | None:
        if not self.commands:
            return None
        if self.index is None:
            self.draft = current_value
            self.index = len(self.commands) - 1
        elif self.index > 0:
            self.index -= 1
        return self.commands[self.index]

    def next(self) -> str | None:
        if self.index is None:
            return None
        next_index = self.index + 1
        if next_index >= len(self.commands):
            draft = self.draft
            self.reset_navigation()
            return draft
        self.index = next_index
        return self.commands[self.index]

    def reset_navigation(self) -> None:
        self.draft = ""
        self.index = None


@dataclass(frozen=True)
class CommandResult:
    kind: Literal["message", "projects", "tasks", "events"]
    message: str | None = None
    projects: ProjectListResponse | None = None
    tasks: TaskListResponse | None = None
    events: EventLogListResponse | None = None


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


def format_event_details(event_details: dict[str, object]) -> str:
    return json.dumps(event_details, sort_keys=True)


def format_project_events_output(events: EventLogListResponse) -> CommandResult:
    if not events:
        return message_result("No events found.")
    return CommandResult(kind="events", events=events)


def run_tui_command(
    command: str,
    fetch_projects_command: Callable[[], ProjectListResponse],
    fetch_tasks_command: Callable[[int], TaskListResponse],
    fetch_project_events_command: Callable[[int], EventLogListResponse],
) -> CommandResult:
    normalized_command = command.strip()
    if normalized_command == "/project list":
        return format_projects_output(fetch_projects_command())

    command_parts = normalized_command.split()
    if command_parts[:3] == ["/project", "event", "list"]:
        if len(command_parts) != 4:
            return message_result("Usage: /project event list PROJECT_ID")

        try:
            project_id = int(command_parts[3])
        except ValueError:
            return message_result("PROJECT_ID must be an integer.")
        return format_project_events_output(fetch_project_events_command(project_id))

    if command_parts[:2] != ["/task", "list"]:
        return message_result(f"Unknown command: {normalized_command or '<empty>'}")
    if len(command_parts) != 3:
        return message_result("Usage: /task list PROJECT_ID")

    try:
        project_id = int(command_parts[2])
    except ValueError:
        return message_result("PROJECT_ID must be an integer.")
    return format_tasks_output(fetch_tasks_command(project_id))


class CommandInput(Input):
    BINDINGS = [
        *Input.BINDINGS,
        Binding("up", "history_previous", "Previous command", show=False),
        Binding("down", "history_next", "Next command", show=False),
    ]

    def action_history_previous(self) -> None:
        app = self.app
        if isinstance(app, AgentbahnTui):
            app.show_previous_command()

    def action_history_next(self) -> None:
        app = self.app
        if isinstance(app, AgentbahnTui):
            app.show_next_command()


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
        fetch_project_events_command: Callable[[int], EventLogListResponse] = fetch_project_events,
    ) -> None:
        super().__init__()
        self._fetch_projects_command = fetch_projects_command
        self._fetch_tasks_command = fetch_tasks_command
        self._fetch_project_events_command = fetch_project_events_command
        self._command_history = CommandHistory(commands=[])
        self._suppressed_history_change_events = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="app-body"):
            with VerticalScroll(id="results-pane"):
                yield Static(get_placeholder_message(), id="message-output")
                yield DataTable(id="results-table")
            yield CommandInput(placeholder="/project list", id="command-input")
        yield Footer()

    def on_mount(self) -> None:
        self._show_message(get_placeholder_message())
        self.query_one("#command-input", Input).focus()

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

    def _show_events(self, events: EventLogListResponse) -> None:
        message_output = self.query_one("#message-output", Static)
        results_table = self.query_one("#results-table", DataTable)
        message_output.display = False
        results_table.display = True
        results_table.clear(columns=True)
        results_table.add_columns("ID", "Entity", "Entity ID", "Event Type", "Details")
        for event in events:
            results_table.add_row(
                str(event.id),
                event.entity_type,
                str(event.entity_id),
                event.event_type,
                format_event_details(event.event_details),
            )

    def _replace_command_input_value(self, value: str) -> None:
        command_input = self.query_one("#command-input", Input)
        self._suppressed_history_change_events += 1
        command_input.value = value
        command_input.cursor_position = len(value)

    def show_previous_command(self) -> None:
        command_input = self.query_one("#command-input", Input)
        previous_command = self._command_history.previous(command_input.value)
        if previous_command is not None:
            self._replace_command_input_value(previous_command)

    def show_next_command(self) -> None:
        next_command = self._command_history.next()
        if next_command is not None:
            self._replace_command_input_value(next_command)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "command-input":
            return
        if self._suppressed_history_change_events > 0:
            self._suppressed_history_change_events -= 1
            return
        if self._command_history.index is not None:
            self._command_history.draft = event.value
            self._command_history.index = None

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._command_history.record(event.value)
        result = run_tui_command(
            event.value,
            self._fetch_projects_command,
            self._fetch_tasks_command,
            self._fetch_project_events_command,
        )
        if result.kind == "projects" and result.projects is not None:
            self._show_projects(result.projects)
        elif result.kind == "tasks" and result.tasks is not None:
            self._show_tasks(result.tasks)
        elif result.kind == "events" and result.events is not None:
            self._show_events(result.events)
        else:
            self._show_message(result.message or get_placeholder_message())
        event.input.value = ""
        event.input.focus()


def run_tui() -> None:
    AgentbahnTui().run()
