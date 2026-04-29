from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from textual.binding import Binding
from textual.app import App
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.containers import VerticalScroll
from textual.widgets import Button
from textual.widgets import DataTable
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import Static

from agentbahn.llms.schemas import LlmConfigLookupResponse
from agentbahn.llms.schemas import LlmConfigResponse
from agentbahn.llms.schemas import LlmConfigUpsertRequest
from agentbahn.projects.schemas import EventLogListResponse
from agentbahn.projects.schemas import ProjectListResponse
from agentbahn.projects.schemas import TaskListResponse
from agentbahn_tui.backend import check_backend_server_running
from agentbahn_tui.command_results import CommandResult
from agentbahn_tui.command_results import message_result
from agentbahn_tui.llms import fetch_llm_config
from agentbahn_tui.llms import save_llm_config
from agentbahn_tui.project_events import fetch_project_events
from agentbahn_tui.projects import fetch_projects
from agentbahn_tui.tasks import fetch_tasks


@dataclass(frozen=True)
class CommandHelpEntry:
    entity: str
    command: str
    shortcut: str
    arguments: str
    description: str


COMMAND_HELP_ENTRIES: tuple[CommandHelpEntry, ...] = (
    CommandHelpEntry(
        entity="project",
        command="/project list",
        shortcut="/pl",
        arguments="-",
        description="List all projects.",
    ),
    CommandHelpEntry(
        entity="project",
        command="/project event list",
        shortcut="/pel",
        arguments="PROJECT_ID",
        description="List event log entries for a project and its related entities.",
    ),
    CommandHelpEntry(
        entity="task",
        command="/task list",
        shortcut="/tl",
        arguments="PROJECT_ID",
        description="List tasks for a project.",
    ),
    CommandHelpEntry(
        entity="llm",
        command="/llm",
        shortcut="-",
        arguments="-",
        description="Show or configure the LLM used by agentbahn.",
    ),
)


def _build_help_table(entries: tuple[CommandHelpEntry, ...]) -> str:
    entity_width = max(len("Entity"), *(len(entry.entity) for entry in entries))
    command_width = max(len("Command"), *(len(entry.command) for entry in entries))
    shortcut_width = max(len("Shortcut"), *(len(entry.shortcut) for entry in entries))
    arguments_width = max(
        len("Arguments"), *(len(entry.arguments) for entry in entries)
    )
    description_width = max(
        len("Description"), *(len(entry.description) for entry in entries)
    )

    def format_row(
        entity: str, command: str, shortcut: str, arguments: str, description: str
    ) -> str:
        return (
            f"{entity.ljust(entity_width)} | "
            f"{command.ljust(command_width)} | "
            f"{shortcut.ljust(shortcut_width)} | "
            f"{arguments.ljust(arguments_width)} | "
            f"{description.ljust(description_width)}"
        ).rstrip()

    separator = "-+-".join(
        [
            "-" * entity_width,
            "-" * command_width,
            "-" * shortcut_width,
            "-" * arguments_width,
            "-" * description_width,
        ]
    )
    rows = [
        "Available commands:",
        "",
        format_row("Entity", "Command", "Shortcut", "Arguments", "Description"),
        separator,
    ]
    rows.extend(
        format_row(
            entry.entity,
            entry.command,
            entry.shortcut,
            entry.arguments,
            entry.description,
        )
        for entry in entries
    )
    rows.extend(
        [
            "",
            "Type a command below and press Enter.",
        ]
    )
    return "\n".join(rows)


def get_placeholder_message() -> str:
    return _build_help_table(COMMAND_HELP_ENTRIES)


def find_agentbahn_home() -> Path:
    return Path.home() / ".agentbahn"


def find_command_history_file() -> Path:
    return find_agentbahn_home() / "command_history"


def load_command_history(history_file: Path) -> list[str]:
    try:
        history_contents = history_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError:
        return []

    commands: list[str] = []
    for line in history_contents.splitlines():
        normalized_command = line.strip()
        if normalized_command:
            commands.append(normalized_command)
    return commands


def append_command_history(history_file: Path, command: str) -> None:
    normalized_command = command.strip()
    if not normalized_command:
        return

    try:
        history_file.parent.mkdir(parents=True, exist_ok=True)
        with history_file.open("a", encoding="utf-8") as history_handle:
            history_handle.write(f"{normalized_command}\n")
    except OSError:
        return


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
    if normalized_command in ("/project list", "/pl"):
        return format_projects_output(fetch_projects_command())

    command_parts = normalized_command.split()
    if command_parts[:3] == ["/project", "event", "list"] or command_parts[:1] == [
        "/pel"
    ]:
        expected_len = 4 if command_parts[0] == "/project" else 2
        id_index = 3 if command_parts[0] == "/project" else 1

        if len(command_parts) != expected_len:
            cmd_str = (
                "/project event list" if command_parts[0] == "/project" else "/pel"
            )
            return message_result(f"Usage: {cmd_str} PROJECT_ID")

        try:
            project_id = int(command_parts[id_index])
        except ValueError:
            return message_result("PROJECT_ID must be an integer.")
        return format_project_events_output(fetch_project_events_command(project_id))

    if command_parts[:2] == ["/task", "list"] or command_parts[:1] == ["/tl"]:
        expected_len = 3 if command_parts[0] == "/task" else 2
        id_index = 2 if command_parts[0] == "/task" else 1

        if len(command_parts) != expected_len:
            cmd_str = "/task list" if command_parts[0] == "/task" else "/tl"
            return message_result(f"Usage: {cmd_str} PROJECT_ID")

        try:
            project_id = int(command_parts[id_index])
        except ValueError:
            return message_result("PROJECT_ID must be an integer.")
        return format_tasks_output(fetch_tasks_command(project_id))

    return message_result(f"Unknown command: {normalized_command or '<empty>'}")


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

    #llm-config-form {
        display: none;
        padding: 1 0;
    }

    #llm-config-form Input {
        margin-bottom: 1;
    }

    #llm-form-status {
        margin-top: 1;
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
        fetch_project_events_command: Callable[
            [int], EventLogListResponse
        ] = fetch_project_events,
        fetch_llm_config_command: Callable[
            [], LlmConfigLookupResponse
        ] = fetch_llm_config,
        save_llm_config_command: Callable[
            [LlmConfigUpsertRequest], LlmConfigResponse
        ] = save_llm_config,
        history_file: Path | None = None,
    ) -> None:
        super().__init__()
        self._fetch_projects_command = fetch_projects_command
        self._fetch_tasks_command = fetch_tasks_command
        self._fetch_project_events_command = fetch_project_events_command
        self._fetch_llm_config_command = fetch_llm_config_command
        self._save_llm_config_command = save_llm_config_command
        self._history_file = history_file or find_command_history_file()
        self._command_history = CommandHistory(
            commands=load_command_history(self._history_file)
        )
        self._suppressed_history_change_events = 0
        self._llm_form_has_existing_config = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="app-body"):
            with VerticalScroll(id="results-pane"):
                yield Static(get_placeholder_message(), id="message-output")
                yield DataTable(id="results-table")
                with Vertical(id="llm-config-form"):
                    yield Label("Provider")
                    yield Input(id="llm-provider-input")
                    yield Label("LLM name")
                    yield Input(id="llm-name-input")
                    yield Label("API key")
                    yield Input(
                        id="llm-api-key-input",
                        password=True,
                        placeholder="Leave blank to keep the existing key",
                    )
                    yield Button("Save LLM configuration", id="llm-save-button")
                    yield Static("", id="llm-form-status")
            yield CommandInput(placeholder="/project list", id="command-input")
        yield Footer()

    def on_mount(self) -> None:
        self._show_message(get_placeholder_message())
        self.query_one("#command-input", Input).focus()

    def _show_message(self, message: str) -> None:
        message_output = self.query_one("#message-output", Static)
        results_table = self.query_one("#results-table", DataTable)
        llm_form = self.query_one("#llm-config-form", Vertical)
        message_output.display = True
        message_output.update(message)
        results_table.display = False
        llm_form.display = False
        results_table.clear(columns=True)

    def _show_projects(self, projects: ProjectListResponse) -> None:
        message_output = self.query_one("#message-output", Static)
        results_table = self.query_one("#results-table", DataTable)
        llm_form = self.query_one("#llm-config-form", Vertical)
        message_output.display = False
        results_table.display = True
        llm_form.display = False
        results_table.clear(columns=True)
        results_table.add_columns("ID", "Name", "Description")
        for project in projects:
            results_table.add_row(str(project.id), project.name, project.description)

    def _show_tasks(self, tasks: TaskListResponse) -> None:
        message_output = self.query_one("#message-output", Static)
        results_table = self.query_one("#results-table", DataTable)
        llm_form = self.query_one("#llm-config-form", Vertical)
        message_output.display = False
        results_table.display = True
        llm_form.display = False
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
        llm_form = self.query_one("#llm-config-form", Vertical)
        message_output.display = False
        results_table.display = True
        llm_form.display = False
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

    def _show_llm_config_form(self) -> None:
        lookup_response = self._fetch_llm_config_command()
        config = lookup_response.config if lookup_response.exists else None

        message_output = self.query_one("#message-output", Static)
        results_table = self.query_one("#results-table", DataTable)
        llm_form = self.query_one("#llm-config-form", Vertical)
        provider_input = self.query_one("#llm-provider-input", Input)
        llm_name_input = self.query_one("#llm-name-input", Input)
        api_key_input = self.query_one("#llm-api-key-input", Input)
        status = self.query_one("#llm-form-status", Static)

        message_output.display = False
        results_table.display = False
        results_table.clear(columns=True)
        llm_form.display = True
        self._llm_form_has_existing_config = config is not None

        provider_input.value = config.provider if config is not None else ""
        llm_name_input.value = config.llm_name if config is not None else ""
        api_key_input.value = ""
        api_key_input.placeholder = (
            "Leave blank to keep the existing key"
            if config is not None and config.api_key_configured
            else "Required"
        )
        status.update(
            "Edit the LLM configuration."
            if config is not None
            else "Create the LLM configuration."
        )
        provider_input.focus()

    def _save_llm_config_form(self) -> None:
        provider_input = self.query_one("#llm-provider-input", Input)
        llm_name_input = self.query_one("#llm-name-input", Input)
        api_key_input = self.query_one("#llm-api-key-input", Input)
        status = self.query_one("#llm-form-status", Static)

        api_key = api_key_input.value.strip()
        if not self._llm_form_has_existing_config and not api_key:
            status.update("API key is required.")
            api_key_input.focus()
            return

        try:
            payload = LlmConfigUpsertRequest(
                provider=provider_input.value,
                llm_name=llm_name_input.value,
                api_key=api_key or None,
            )
        except ValueError as exc:
            status.update(str(exc))
            return

        saved_config = self._save_llm_config_command(payload)
        self._llm_form_has_existing_config = True
        api_key_input.value = ""
        api_key_input.placeholder = "Leave blank to keep the existing key"
        status.update(
            "Saved LLM configuration.\n"
            f"Provider: {saved_config.provider}\n"
            f"LLM name: {saved_config.llm_name}\n"
            "API key: "
            f"{'configured' if saved_config.api_key_configured else 'missing'}"
        )

    def _replace_command_input_value(self, value: str) -> None:
        command_input = self.query_one("#command-input", Input)
        self._suppressed_history_change_events += 1
        command_input.value = value
        command_input.cursor_position = len(value)

    def _set_command_input_secret_mode(self, enabled: bool) -> None:
        command_input = self.query_one("#command-input", Input)
        command_input.password = enabled

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
        if event.input.id != "command-input":
            if event.input.id in {
                "llm-provider-input",
                "llm-name-input",
                "llm-api-key-input",
            }:
                self._save_llm_config_form()
            return

        normalized_value = event.value.strip()
        if normalized_value == "/llm":
            self._command_history.record(event.value)
            append_command_history(self._history_file, event.value)
            self._set_command_input_secret_mode(False)
            self._show_llm_config_form()
            event.input.value = ""
            return
        else:
            self._command_history.record(event.value)
            append_command_history(self._history_file, event.value)
            self._set_command_input_secret_mode(False)
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

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "llm-save-button":
            self._save_llm_config_form()


def run_tui(
    backend_check: Callable[[], None] = check_backend_server_running,
) -> None:
    backend_check()
    AgentbahnTui().run()
