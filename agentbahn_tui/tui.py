from __future__ import annotations

import json
from collections.abc import Callable
from collections.abc import Iterator
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
from textual.widgets import Select
from textual.widgets import Static

from agentbahn.codebase_agent.schemas import CodebaseAgentStreamEvent
from agentbahn.llms.schemas import LlmConfigListResponse
from agentbahn.llms.schemas import LlmConfigResponse
from agentbahn.llms.schemas import LlmConfigUpsertRequest
from agentbahn.projects.schemas import EventLogListResponse
from agentbahn.projects.schemas import ProjectListResponse
from agentbahn.projects.schemas import TaskListResponse
from agentbahn_tui.agents import stream_codebase_agent
from agentbahn_tui.backend import check_backend_server_running
from agentbahn_tui.command_results import CommandResult
from agentbahn_tui.command_results import message_result
from agentbahn_tui.llms import fetch_llm_configs
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
        entity="help",
        command="/help",
        shortcut="/h",
        arguments="-",
        description="Show this help menu.",
    ),
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
    CommandHelpEntry(
        entity="model",
        command="/model",
        shortcut="-",
        arguments="-",
        description="Pick the LLM configuration used by codebase-agent chat.",
    ),
)

SLASH_COMMAND_WORDS: frozenset[str] = frozenset(
    {
        "/h",
        "/help",
        "/llm",
        "/model",
        "/pel",
        "/pl",
        "/project",
        "/task",
        "/tl",
    }
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
            "Messages that do not start with / are sent to the DefaultAgent.",
            "Type a command or message below and press Enter.",
        ]
    )
    return "\n".join(rows)


def get_placeholder_message() -> str:
    return _build_help_table(COMMAND_HELP_ENTRIES)


def is_registered_slash_command(command: str) -> bool:
    command_parts = command.strip().split(maxsplit=1)
    if not command_parts:
        return False
    return command_parts[0] in SLASH_COMMAND_WORDS


def find_agentbahn_home() -> Path:
    return Path.home() / ".agentbahn"


def find_command_history_file() -> Path:
    return find_agentbahn_home() / "command_history"


def find_model_config_file() -> Path:
    return find_agentbahn_home() / "model_config"


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


def load_selected_model_config_id(model_config_file: Path) -> int | None:
    try:
        model_config_contents = model_config_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None

    normalized_config_id = model_config_contents.strip()
    if not normalized_config_id:
        return None

    try:
        return int(normalized_config_id)
    except ValueError:
        return None


def save_selected_model_config_id(model_config_file: Path, config_id: int) -> None:
    try:
        model_config_file.parent.mkdir(parents=True, exist_ok=True)
        model_config_file.write_text(f"{config_id}\n", encoding="utf-8")
    except OSError:
        return


def load_verified_selected_model_config_id(
    model_config_file: Path,
    fetch_llm_configs_command: Callable[[], LlmConfigListResponse],
) -> int | None:
    config_id = load_selected_model_config_id(model_config_file)
    if config_id is None:
        return None

    configs = fetch_llm_configs_command().configs
    if any(config.id == config_id for config in configs):
        return config_id
    return None


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


def _format_llm_config_list(configs: list[LlmConfigResponse]) -> str:
    if not configs:
        return "Existing LLM configurations: none"
    lines = ["Existing LLM configurations:"]
    for config in configs:
        api_key_status = "configured" if config.api_key_configured else "missing"
        lines.append(
            f"{config.id}. {config.name} - {config.provider}/{config.llm_name} "
            f"({config.lm_backend_path}, API key {api_key_status})"
        )
    return "\n".join(lines)


def _format_llm_config_option(config: LlmConfigResponse) -> str:
    api_key_status = "configured" if config.api_key_configured else "missing"
    return (
        f"{config.id}. {config.name} - {config.provider}/{config.llm_name} "
        f"({config.lm_backend_path}, API key {api_key_status})"
    )


def _format_model_selection(config: LlmConfigResponse | None) -> str:
    if config is None:
        return "No model selected. Chat will use the backend default LLM configuration."
    return (
        "Selected model for codebase-agent chat:\n"
        f"ID: {config.id}\n"
        f"Name: {config.name}\n"
        f"Provider: {config.provider}\n"
        f"LLM name: {config.llm_name}\n"
        f"LM backend path: {config.lm_backend_path}"
    )


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
    if normalized_command in ("/help", "/h"):
        return message_result(get_placeholder_message())

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

    #llm-config-form Select {
        margin-bottom: 1;
    }

    #llm-backend-path-tip {
        margin-bottom: 1;
        color: $text-muted;
    }

    #llm-form-status {
        margin-top: 1;
    }

    #model-select-form {
        display: none;
        padding: 1 0;
    }

    #model-select-form Select {
        margin-bottom: 1;
    }

    #model-config-table {
        height: auto;
        margin-bottom: 1;
    }

    #model-status {
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
        fetch_llm_configs_command: Callable[
            [], LlmConfigListResponse
        ] = fetch_llm_configs,
        save_llm_config_command: Callable[
            [LlmConfigUpsertRequest], LlmConfigResponse
        ] = save_llm_config,
        stream_agent_command: Callable[
            [str, int | None], Iterator[CodebaseAgentStreamEvent]
        ] = stream_codebase_agent,
        history_file: Path | None = None,
    ) -> None:
        super().__init__()
        self._fetch_projects_command = fetch_projects_command
        self._fetch_tasks_command = fetch_tasks_command
        self._fetch_project_events_command = fetch_project_events_command
        self._fetch_llm_configs_command = fetch_llm_configs_command
        self._save_llm_config_command = save_llm_config_command
        self._stream_agent_command = stream_agent_command
        self._history_file = history_file or find_command_history_file()
        self._model_config_file = (
            self._history_file.parent / find_model_config_file().name
            if history_file is not None
            else find_model_config_file()
        )
        self._command_history = CommandHistory(
            commands=load_command_history(self._history_file)
        )
        self._suppressed_history_change_events = 0
        self._agent_output = ""
        self._llm_configs: list[LlmConfigResponse] = []
        self._selected_llm_config_id: int | None = None
        self._suppress_llm_select_change = False
        self._selected_model_config_id = load_verified_selected_model_config_id(
            self._model_config_file,
            self._fetch_llm_configs_command,
        )

    def _set_llm_form_controls_disabled(self, disabled: bool) -> None:
        self.query_one("#llm-config-select", Select).disabled = disabled
        self.query_one("#llm-friendly-name-input", Input).disabled = disabled
        self.query_one("#llm-provider-input", Input).disabled = disabled
        self.query_one("#llm-name-input", Input).disabled = disabled
        self.query_one("#llm-backend-path-input", Input).disabled = disabled
        self.query_one("#llm-api-key-input", Input).disabled = disabled
        self.query_one("#llm-save-button", Button).disabled = disabled

    def _set_model_form_controls_disabled(self, disabled: bool) -> None:
        self.query_one("#model-config-select", Select).disabled = disabled
        self.query_one("#model-use-button", Button).disabled = disabled

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="app-body"):
            with VerticalScroll(id="results-pane"):
                yield Static(get_placeholder_message(), id="message-output")
                yield DataTable(id="results-table")
                with Vertical(id="llm-config-form"):
                    yield Label("Configuration")
                    yield Select(
                        [("Create new configuration", "new")],
                        id="llm-config-select",
                        allow_blank=False,
                        disabled=True,
                    )
                    yield Static("", id="llm-config-list")
                    yield Label("Name")
                    yield Input(id="llm-friendly-name-input", disabled=True)
                    yield Label("Provider")
                    yield Input(id="llm-provider-input", disabled=True)
                    yield Label("LLM name")
                    yield Input(id="llm-name-input", disabled=True)
                    yield Label("LM backend path")
                    yield Input(id="llm-backend-path-input", disabled=True)
                    yield Static(
                        "Import path for a custom DSPy LM backend module. "
                        "'default' uses DSPy's standard LM backend for the "
                        "selected provider and model.",
                        id="llm-backend-path-tip",
                    )
                    yield Label("API key")
                    yield Input(
                        id="llm-api-key-input",
                        password=True,
                        placeholder="Required",
                        disabled=True,
                    )
                    yield Button(
                        "Create LLM configuration",
                        id="llm-save-button",
                        disabled=True,
                    )
                    yield Static("", id="llm-form-status")
                with Vertical(id="model-select-form"):
                    yield Label("Model")
                    yield Select(
                        [],
                        id="model-config-select",
                        allow_blank=True,
                        disabled=True,
                    )
                    yield DataTable(id="model-config-table")
                    yield Button(
                        "Use selected model",
                        id="model-use-button",
                        disabled=True,
                    )
                    yield Static("", id="model-status")
            yield CommandInput(placeholder="/project list", id="command-input")
        yield Footer()

    def on_mount(self) -> None:
        self._show_message(get_placeholder_message())
        self.query_one("#command-input", Input).focus()

    def _show_message(self, message: str) -> None:
        message_output = self.query_one("#message-output", Static)
        results_table = self.query_one("#results-table", DataTable)
        llm_form = self.query_one("#llm-config-form", Vertical)
        model_form = self.query_one("#model-select-form", Vertical)
        message_output.display = True
        message_output.update(message)
        results_table.display = False
        llm_form.display = False
        model_form.display = False
        self._set_llm_form_controls_disabled(True)
        self._set_model_form_controls_disabled(True)
        results_table.clear(columns=True)

    def _start_agent_stream(self, query: str) -> None:
        self._agent_output = ""
        self._show_message("DefaultAgent is running...\n")
        self.run_worker(
            lambda: self._run_agent_stream(query),
            thread=True,
            exclusive=True,
            group="agent",
        )

    def _run_agent_stream(self, query: str) -> None:
        try:
            for event in self._stream_agent_command(
                query,
                self._selected_model_config_id,
            ):
                self.call_from_thread(self._apply_agent_stream_event, event)
        except Exception as exc:
            self.call_from_thread(
                self._append_agent_output,
                f"\nAgent request failed: {exc}",
            )

    def _apply_agent_stream_event(self, event: CodebaseAgentStreamEvent) -> None:
        if event.type == "token" and event.content:
            self._append_agent_output(event.content)
        elif event.type == "result" and event.content:
            if self._agent_output and not self._agent_output.endswith("\n"):
                self._append_agent_output("\n")
            self._append_agent_output(event.content)
        elif event.type == "error":
            detail = event.detail or "Agent request failed."
            self._append_agent_output(f"\n{detail}")

    def _append_agent_output(self, content: str) -> None:
        self._agent_output += content
        self._show_message(self._agent_output)

    def _show_projects(self, projects: ProjectListResponse) -> None:
        message_output = self.query_one("#message-output", Static)
        results_table = self.query_one("#results-table", DataTable)
        llm_form = self.query_one("#llm-config-form", Vertical)
        model_form = self.query_one("#model-select-form", Vertical)
        message_output.display = False
        results_table.display = True
        llm_form.display = False
        model_form.display = False
        self._set_llm_form_controls_disabled(True)
        self._set_model_form_controls_disabled(True)
        results_table.clear(columns=True)
        results_table.add_columns("ID", "Name", "Description")
        for project in projects:
            results_table.add_row(str(project.id), project.name, project.description)

    def _show_tasks(self, tasks: TaskListResponse) -> None:
        message_output = self.query_one("#message-output", Static)
        results_table = self.query_one("#results-table", DataTable)
        llm_form = self.query_one("#llm-config-form", Vertical)
        model_form = self.query_one("#model-select-form", Vertical)
        message_output.display = False
        results_table.display = True
        llm_form.display = False
        model_form.display = False
        self._set_llm_form_controls_disabled(True)
        self._set_model_form_controls_disabled(True)
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
        model_form = self.query_one("#model-select-form", Vertical)
        message_output.display = False
        results_table.display = True
        llm_form.display = False
        model_form.display = False
        self._set_llm_form_controls_disabled(True)
        self._set_model_form_controls_disabled(True)
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
        configs = self._fetch_llm_configs_command().configs
        self._llm_configs = configs
        self._selected_llm_config_id = None

        message_output = self.query_one("#message-output", Static)
        results_table = self.query_one("#results-table", DataTable)
        llm_form = self.query_one("#llm-config-form", Vertical)
        model_form = self.query_one("#model-select-form", Vertical)
        llm_config_select = self.query_one("#llm-config-select", Select)
        llm_config_list = self.query_one("#llm-config-list", Static)
        friendly_name_input = self.query_one("#llm-friendly-name-input", Input)
        provider_input = self.query_one("#llm-provider-input", Input)
        llm_name_input = self.query_one("#llm-name-input", Input)
        lm_backend_path_input = self.query_one("#llm-backend-path-input", Input)
        api_key_input = self.query_one("#llm-api-key-input", Input)
        save_button = self.query_one("#llm-save-button", Button)
        status = self.query_one("#llm-form-status", Static)

        message_output.display = False
        results_table.display = False
        results_table.clear(columns=True)
        llm_form.display = True
        model_form.display = False
        self._set_llm_form_controls_disabled(False)
        self._set_model_form_controls_disabled(True)

        llm_config_select.set_options(
            [
                ("Create new configuration", "new"),
                *[
                    (_format_llm_config_option(config), str(config.id))
                    for config in configs
                ],
            ]
        )
        llm_config_select.value = "new"
        llm_config_list.update(_format_llm_config_list(configs))
        friendly_name_input.value = ""
        provider_input.value = ""
        llm_name_input.value = ""
        lm_backend_path_input.value = "default"
        api_key_input.value = ""
        api_key_input.placeholder = "Required"
        save_button.label = "Create LLM configuration"
        status.update("Create a new LLM configuration.")
        friendly_name_input.focus()

    def _show_model_select_form(self) -> None:
        configs = self._fetch_llm_configs_command().configs
        self._llm_configs = configs

        message_output = self.query_one("#message-output", Static)
        results_table = self.query_one("#results-table", DataTable)
        llm_form = self.query_one("#llm-config-form", Vertical)
        model_form = self.query_one("#model-select-form", Vertical)
        model_config_select = self.query_one("#model-config-select", Select)
        model_config_table = self.query_one("#model-config-table", DataTable)
        status = self.query_one("#model-status", Static)

        message_output.display = False
        results_table.display = False
        results_table.clear(columns=True)
        llm_form.display = False
        model_form.display = True
        self._set_llm_form_controls_disabled(True)

        if not configs:
            model_config_select.set_options([])
            model_config_table.clear(columns=True)
            model_config_table.add_columns("Selected", "ID", "Name", "Provider", "LLM")
            status.update("No LLM configurations found. Use /llm to create one.")
            self._set_model_form_controls_disabled(True)
            return

        model_config_select.set_options(
            [(_format_llm_config_option(config), str(config.id)) for config in configs]
        )
        selected_config = next(
            (
                config
                for config in configs
                if config.id == self._selected_model_config_id
            ),
            configs[0],
        )
        model_config_select.value = str(selected_config.id)
        self._refresh_model_config_table(self._selected_model_config_id)
        status.update(
            _format_model_selection(
                selected_config
                if self._selected_model_config_id == selected_config.id
                else None
            )
        )
        self._set_model_form_controls_disabled(False)
        model_config_select.focus()

    def _refresh_model_config_table(self, selected_config_id: int | None) -> None:
        model_config_table = self.query_one("#model-config-table", DataTable)
        model_config_table.clear(columns=True)
        model_config_table.add_columns("Selected", "ID", "Name", "Provider", "LLM")
        for config in self._llm_configs:
            selected_marker = "✓" if config.id == selected_config_id else ""
            model_config_table.add_row(
                selected_marker,
                str(config.id),
                config.name,
                config.provider,
                config.llm_name,
            )

    def _select_model_config(self) -> None:
        model_config_select = self.query_one("#model-config-select", Select)
        status = self.query_one("#model-status", Static)
        if model_config_select.value == Select.NULL:
            status.update("Select an LLM configuration.")
            return

        config_id = int(str(model_config_select.value))
        selected_config = next(
            (config for config in self._llm_configs if config.id == config_id),
            None,
        )
        if selected_config is None:
            status.update(f"LLM configuration {config_id} was not found.")
            return

        self._selected_model_config_id = selected_config.id
        save_selected_model_config_id(self._model_config_file, selected_config.id)
        self._refresh_model_config_table(selected_config.id)
        status.update(_format_model_selection(selected_config))

    def _select_llm_config(self, config_id: int | None) -> None:
        self._selected_llm_config_id = config_id
        provider_input = self.query_one("#llm-provider-input", Input)
        friendly_name_input = self.query_one("#llm-friendly-name-input", Input)
        llm_name_input = self.query_one("#llm-name-input", Input)
        lm_backend_path_input = self.query_one("#llm-backend-path-input", Input)
        api_key_input = self.query_one("#llm-api-key-input", Input)
        save_button = self.query_one("#llm-save-button", Button)
        status = self.query_one("#llm-form-status", Static)

        if config_id is None:
            friendly_name_input.value = ""
            provider_input.value = ""
            llm_name_input.value = ""
            lm_backend_path_input.value = "default"
            api_key_input.value = ""
            api_key_input.placeholder = "Required"
            save_button.label = "Create LLM configuration"
            status.update("Create a new LLM configuration.")
            friendly_name_input.focus()
            return

        selected_config = next(
            (config for config in self._llm_configs if config.id == config_id),
            None,
        )
        if selected_config is None:
            status.update(f"LLM configuration {config_id} was not found.")
            return

        friendly_name_input.value = selected_config.name
        provider_input.value = selected_config.provider
        llm_name_input.value = selected_config.llm_name
        lm_backend_path_input.value = selected_config.lm_backend_path
        api_key_input.value = ""
        api_key_input.placeholder = (
            "Leave blank to keep configured key"
            if selected_config.api_key_configured
            else "Required"
        )
        save_button.label = "Update LLM configuration"
        status.update(f"Editing LLM configuration {selected_config.id}.")
        friendly_name_input.focus()

    def _save_llm_config_form(self) -> None:
        friendly_name_input = self.query_one("#llm-friendly-name-input", Input)
        provider_input = self.query_one("#llm-provider-input", Input)
        llm_name_input = self.query_one("#llm-name-input", Input)
        lm_backend_path_input = self.query_one("#llm-backend-path-input", Input)
        api_key_input = self.query_one("#llm-api-key-input", Input)
        status = self.query_one("#llm-form-status", Static)

        api_key = api_key_input.value.strip()
        if self._selected_llm_config_id is None and not api_key:
            status.update("API key is required.")
            api_key_input.focus()
            return

        try:
            payload = LlmConfigUpsertRequest(
                id=self._selected_llm_config_id,
                name=friendly_name_input.value,
                provider=provider_input.value,
                llm_name=llm_name_input.value,
                lm_backend_path=lm_backend_path_input.value,
                api_key=api_key or None,
            )
        except ValueError as exc:
            status.update(str(exc))
            return

        saved_config = self._save_llm_config_command(payload)
        configs = self._fetch_llm_configs_command().configs
        if not any(config.id == saved_config.id for config in configs):
            configs = sorted(
                [*configs, saved_config],
                key=lambda config: config.id,
            )
        self._llm_configs = configs
        self._selected_llm_config_id = saved_config.id
        llm_config_list = self.query_one("#llm-config-list", Static)
        llm_config_select = self.query_one("#llm-config-select", Select)
        save_button = self.query_one("#llm-save-button", Button)
        api_key_input.value = ""
        friendly_name_input.value = saved_config.name
        provider_input.value = saved_config.provider
        llm_name_input.value = saved_config.llm_name
        lm_backend_path_input.value = saved_config.lm_backend_path
        api_key_input.placeholder = (
            "Leave blank to keep configured key"
            if saved_config.api_key_configured
            else "Required"
        )
        self._suppress_llm_select_change = True
        llm_config_select.set_options(
            [
                ("Create new configuration", "new"),
                *[
                    (_format_llm_config_option(config), str(config.id))
                    for config in configs
                ],
            ]
        )
        llm_config_select.value = str(saved_config.id)
        self.set_timer(
            0.1,
            lambda: setattr(self, "_suppress_llm_select_change", False),
        )
        llm_config_list.update(_format_llm_config_list(configs))
        save_button.label = "Update LLM configuration"
        action = "Updated" if payload.id is not None else "Created"
        status.update(
            f"{action} LLM configuration.\n"
            f"ID: {saved_config.id}\n"
            f"Name: {saved_config.name}\n"
            f"Provider: {saved_config.provider}\n"
            f"LLM name: {saved_config.llm_name}\n"
            f"LM backend path: {saved_config.lm_backend_path}\n"
            "API key: "
            f"{'configured' if saved_config.api_key_configured else 'missing'}"
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "model-config-select":
            if event.value == Select.NULL:
                return
            selected_config = next(
                (
                    config
                    for config in self._llm_configs
                    if config.id == int(str(event.value))
                ),
                None,
            )
            self.query_one("#model-status", Static).update(
                _format_model_selection(selected_config)
            )
            return

        if event.select.id != "llm-config-select":
            return
        if self._suppress_llm_select_change:
            return
        if event.value == "new" or event.value == Select.NULL:
            self._select_llm_config(None)
            return
        self._select_llm_config(int(str(event.value)))

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
                "llm-friendly-name-input",
                "llm-provider-input",
                "llm-name-input",
                "llm-backend-path-input",
                "llm-api-key-input",
            }:
                self._save_llm_config_form()
            return

        normalized_value = event.value.strip()
        if not normalized_value:
            self._show_message(get_placeholder_message())
            event.input.value = ""
            event.input.focus()
            return

        if normalized_value == "/llm":
            self._command_history.record(event.value)
            append_command_history(self._history_file, event.value)
            self._set_command_input_secret_mode(False)
            self._show_llm_config_form()
            event.input.value = ""
            return
        if normalized_value == "/model":
            self._command_history.record(event.value)
            append_command_history(self._history_file, event.value)
            self._set_command_input_secret_mode(False)
            self._show_model_select_form()
            event.input.value = ""
            return
        if is_registered_slash_command(normalized_value):
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
            return

        if normalized_value.startswith("/"):
            self._command_history.record(event.value)
            append_command_history(self._history_file, event.value)
            self._set_command_input_secret_mode(False)
            self._show_message(f"Unknown command: {normalized_value}")
            event.input.value = ""
            event.input.focus()
            return

        self._command_history.record(event.value)
        append_command_history(self._history_file, event.value)
        self._set_command_input_secret_mode(False)
        event.input.value = ""
        event.input.focus()
        self._start_agent_stream(normalized_value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "llm-save-button":
            self._save_llm_config_form()
        elif event.button.id == "model-use-button":
            self._select_model_config()


def run_tui(
    backend_check: Callable[[], None] = check_backend_server_running,
) -> None:
    backend_check()
    AgentbahnTui().run()
