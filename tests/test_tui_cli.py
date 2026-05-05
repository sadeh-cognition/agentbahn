from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from agentbahn.projects.schemas import EventLogResponse
from agentbahn.projects.schemas import FeatureResponse
from django.core.management import call_command

from agentbahn.codebase_agent.schemas import CodebaseAgentStreamEvent
from agentbahn_tui.project_events import FEATURE_EVENT_ENTITY_TYPE
from agentbahn_tui.project_events import PROJECT_EVENT_ENTITY_TYPE
from agentbahn_tui.project_events import TASK_EVENT_ENTITY_TYPE
from agentbahn_tui.command_results import CommandResult
from agentbahn_tui.project_events import fetch_project_events
from agentbahn.projects.schemas import TaskResponse
from agentbahn_tui import cli
from agentbahn_tui.backend import BackendUnavailableError
from agentbahn_tui.tui import run_tui
from agentbahn_tui.tui import AgentbahnTui
from agentbahn_tui.tui import append_command_history
from agentbahn_tui.tui import CommandHistory
from agentbahn_tui.tui import format_event_details
from agentbahn_tui.tui import format_tasks_output
from agentbahn_tui.tui import get_placeholder_message
from agentbahn_tui.tui import load_command_history
from agentbahn_tui.tui import run_tui_command


def test_placeholder_message_is_stable() -> None:
    assert (
        get_placeholder_message() == "Available commands:\n"
        "\n"
        "Entity  | Command             | Shortcut | Arguments  | Description\n"
        "--------+---------------------+----------+------------+---------------------------------------------------------------\n"
        "project | /project list       | /pl      | -          | List all projects.\n"
        "project | /project event list | /pel     | PROJECT_ID | List event log entries for a project and its related entities.\n"
        "task    | /task list          | /tl      | PROJECT_ID | List tasks for a project.\n"
        "llm     | /llm                | -        | -          | Show or configure the LLM used by agentbahn.\n"
        "\n"
        "Messages that do not start with / are sent to the DefaultAgent.\n"
        "Type a command or message below and press Enter."
    )


def test_management_command_starts_tui(monkeypatch) -> None:
    calls: list[str] = []

    def fake_run_tui() -> None:
        calls.append("started")

    monkeypatch.setattr(
        "agentbahn_tui.management.commands.start_tui.run_tui", fake_run_tui
    )

    call_command("start_tui")

    assert calls == ["started"]


def test_run_tui_checks_backend_before_starting_app(monkeypatch) -> None:
    calls: list[str] = []

    def fake_backend_check() -> None:
        calls.append("checked")

    def fake_app_run(self) -> None:
        calls.append("started")

    monkeypatch.setattr(AgentbahnTui, "run", fake_app_run)

    run_tui(backend_check=fake_backend_check)

    assert calls == ["checked", "started"]


def test_run_tui_raises_when_backend_check_fails(monkeypatch) -> None:
    calls: list[str] = []

    def fake_backend_check() -> None:
        calls.append("checked")
        raise BackendUnavailableError("backend unavailable")

    def fake_app_run(self) -> None:
        calls.append("started")

    monkeypatch.setattr(AgentbahnTui, "run", fake_app_run)

    try:
        run_tui(backend_check=fake_backend_check)
    except BackendUnavailableError as exc:
        assert str(exc) == "backend unavailable"
    else:
        raise AssertionError("Expected BackendUnavailableError")

    assert calls == ["checked"]


def test_console_script_dispatches_to_management_command(monkeypatch) -> None:
    executed_argv: list[str] = []

    def fake_execute_from_command_line(argv: Sequence[str]) -> None:
        executed_argv.extend(argv)

    monkeypatch.delenv("DJANGO_SETTINGS_MODULE", raising=False)
    monkeypatch.setattr(
        cli, "execute_from_command_line", fake_execute_from_command_line
    )

    exit_code = cli.main(["--help"])

    assert exit_code == 0
    assert executed_argv == ["agentbahn-tui", "start_tui", "--help"]
    assert cli.os.environ["DJANGO_SETTINGS_MODULE"] == "agentbahn.settings"


def test_run_tui_command_lists_tasks_for_project() -> None:
    expected_tasks = [
        TaskResponse(
            id=11,
            project_id=7,
            project_name="Roadmap",
            feature_id=3,
            feature_name="CLI",
            user_id=5,
            user_username="alice",
            title="List tasks",
            description="Show tasks for a project",
            status="todo",
            date_created="2026-04-19T10:00:00Z",
            date_updated="2026-04-19T10:30:00Z",
        )
    ]

    def fetch_tasks_cmd(project_id):
        return expected_tasks

    output = run_tui_command(
        "/task list 7",
        fetch_projects_command=lambda: [],
        fetch_tasks_command=fetch_tasks_cmd,
        fetch_project_events_command=lambda _project_id: [],
    )

    assert output == CommandResult(kind="tasks", tasks=expected_tasks)

    shortcut_output = run_tui_command(
        "/tl 7",
        fetch_projects_command=lambda: [],
        fetch_tasks_command=fetch_tasks_cmd,
        fetch_project_events_command=lambda _project_id: [],
    )

    assert shortcut_output == CommandResult(kind="tasks", tasks=expected_tasks)


def test_run_tui_command_validates_task_list_arguments() -> None:
    assert run_tui_command(
        "/task list",
        fetch_projects_command=lambda: [],
        fetch_tasks_command=lambda _project_id: [],
        fetch_project_events_command=lambda _project_id: [],
    ) == CommandResult(kind="message", message="Usage: /task list PROJECT_ID")
    assert run_tui_command(
        "/tl",
        fetch_projects_command=lambda: [],
        fetch_tasks_command=lambda _project_id: [],
        fetch_project_events_command=lambda _project_id: [],
    ) == CommandResult(kind="message", message="Usage: /tl PROJECT_ID")
    assert run_tui_command(
        "/task list abc",
        fetch_projects_command=lambda: [],
        fetch_tasks_command=lambda _project_id: [],
        fetch_project_events_command=lambda _project_id: [],
    ) == CommandResult(kind="message", message="PROJECT_ID must be an integer.")


def test_run_tui_command_lists_project_events() -> None:
    expected_events = [
        EventLogResponse(
            id=21,
            entity_type="task",
            entity_id=13,
            event_type="task.updated",
            event_details={"project_id": 7, "status": "done"},
        )
    ]

    def fetch_project_events_cmd(project_id):
        return expected_events

    output = run_tui_command(
        "/project event list 7",
        fetch_projects_command=lambda: [],
        fetch_tasks_command=lambda _project_id: [],
        fetch_project_events_command=fetch_project_events_cmd,
    )

    assert output == CommandResult(kind="events", events=expected_events)

    shortcut_output = run_tui_command(
        "/pel 7",
        fetch_projects_command=lambda: [],
        fetch_tasks_command=lambda _project_id: [],
        fetch_project_events_command=fetch_project_events_cmd,
    )

    assert shortcut_output == CommandResult(kind="events", events=expected_events)


def test_run_tui_command_validates_project_event_arguments() -> None:
    assert run_tui_command(
        "/project event list",
        fetch_projects_command=lambda: [],
        fetch_tasks_command=lambda _project_id: [],
        fetch_project_events_command=lambda _project_id: [],
    ) == CommandResult(kind="message", message="Usage: /project event list PROJECT_ID")
    assert run_tui_command(
        "/pel",
        fetch_projects_command=lambda: [],
        fetch_tasks_command=lambda _project_id: [],
        fetch_project_events_command=lambda _project_id: [],
    ) == CommandResult(kind="message", message="Usage: /pel PROJECT_ID")
    assert run_tui_command(
        "/project event list abc",
        fetch_projects_command=lambda: [],
        fetch_tasks_command=lambda _project_id: [],
        fetch_project_events_command=lambda _project_id: [],
    ) == CommandResult(kind="message", message="PROJECT_ID must be an integer.")


def test_format_tasks_output_handles_empty_results() -> None:
    assert format_tasks_output([]) == CommandResult(
        kind="message", message="No tasks found."
    )


def test_fetch_project_events_includes_project_feature_and_task_events() -> None:
    events = fetch_project_events(
        7,
        fetch_features_command=lambda project_id: [
            FeatureResponse(
                id=3,
                project_id=project_id or 0,
                parent_feature_id=None,
                name="CLI",
                description="Command interface",
                date_created="2026-04-19T10:00:00Z",
                date_updated="2026-04-19T10:30:00Z",
            )
        ],
        fetch_tasks_command=lambda project_id: [
            TaskResponse(
                id=11,
                project_id=project_id,
                project_name="Roadmap",
                feature_id=3,
                feature_name="CLI",
                user_id=5,
                user_username="alice",
                title="List events",
                description="Show project events",
                status="todo",
                date_created="2026-04-19T10:00:00Z",
                date_updated="2026-04-19T10:30:00Z",
            )
        ],
        fetch_event_logs_command=lambda entity_type, entity_id: [
            EventLogResponse(
                id=entity_id + 100,
                entity_type=entity_type,
                entity_id=entity_id,
                event_type=f"{entity_type}.updated",
                event_details={"entity_id": entity_id},
            )
        ],
    )

    assert [(event.id, event.entity_type, event.entity_id) for event in events] == [
        (111, TASK_EVENT_ENTITY_TYPE, 11),
        (107, PROJECT_EVENT_ENTITY_TYPE, 7),
        (103, FEATURE_EVENT_ENTITY_TYPE, 3),
    ]


def test_format_event_details_sorts_keys_for_stable_display() -> None:
    assert format_event_details({"b": 2, "a": 1}) == '{"a": 1, "b": 2}'


def test_fetch_project_events_uses_backend_entity_type_values() -> None:
    requested_entity_keys: list[tuple[str, int]] = []

    fetch_project_events(
        4,
        fetch_features_command=lambda project_id: [
            FeatureResponse(
                id=8,
                project_id=project_id or 0,
                parent_feature_id=None,
                name="Events",
                description="Project event list",
                date_created="2026-04-19T10:00:00Z",
                date_updated="2026-04-19T10:30:00Z",
            )
        ],
        fetch_tasks_command=lambda project_id: [
            TaskResponse(
                id=15,
                project_id=project_id,
                project_name="Roadmap",
                feature_id=8,
                feature_name="Events",
                user_id=2,
                user_username="alice",
                title="Inspect logs",
                description="Inspect project event logs",
                status="todo",
                date_created="2026-04-19T10:00:00Z",
                date_updated="2026-04-19T10:30:00Z",
            )
        ],
        fetch_event_logs_command=lambda entity_type, entity_id: (
            requested_entity_keys.append((entity_type, entity_id)) or []
        ),
    )

    assert requested_entity_keys == [
        (PROJECT_EVENT_ENTITY_TYPE, 4),
        (FEATURE_EVENT_ENTITY_TYPE, 8),
        (TASK_EVENT_ENTITY_TYPE, 15),
    ]


def test_command_history_navigates_backwards_and_restores_draft() -> None:
    history = CommandHistory(commands=[])

    history.record("/project list")
    history.record("/task list 7")

    assert history.previous("") == "/task list 7"
    assert history.previous("") == "/project list"
    assert history.next() == "/task list 7"
    assert history.next() == ""
    assert history.next() is None


def test_command_history_file_loads_non_empty_commands(tmp_path) -> None:
    history_file = tmp_path / "command_history"
    history_file.write_text("/project list\n\n /task list 7 \n", encoding="utf-8")

    assert load_command_history(history_file) == ["/project list", "/task list 7"]


def test_command_history_file_appends_commands(tmp_path) -> None:
    history_file = tmp_path / "state" / "command_history"

    append_command_history(history_file, "/project list")
    append_command_history(history_file, "   ")
    append_command_history(history_file, " /task list 7 ")

    assert history_file.read_text(encoding="utf-8") == "/project list\n/task list 7\n"


def test_tui_command_history_uses_up_and_down_keys(tmp_path) -> None:
    async def run_test() -> None:
        history_file = tmp_path / "command_history"
        app = AgentbahnTui(
            fetch_projects_command=lambda: [],
            fetch_tasks_command=lambda _project_id: [],
            fetch_project_events_command=lambda _project_id: [],
            history_file=history_file,
        )

        async with app.run_test() as pilot:
            await pilot.press(
                "/", "p", "r", "o", "j", "e", "c", "t", " ", "l", "i", "s", "t"
            )
            await pilot.press("enter")
            await pilot.press(
                "/", "t", "a", "s", "k", " ", "l", "i", "s", "t", " ", "7"
            )
            await pilot.press("enter")
            await pilot.press("up")
            assert app.query_one("#command-input").value == "/task list 7"
            await pilot.press("up")
            assert app.query_one("#command-input").value == "/project list"
            await pilot.press("down")
            assert app.query_one("#command-input").value == "/task list 7"
            await pilot.press("ctrl+u")
            await pilot.press("/", "d", "r", "a", "f", "t")
            await pilot.press("up")
            assert app.query_one("#command-input").value == "/task list 7"
            await pilot.press("down")
            assert app.query_one("#command-input").value == "/draft"

    asyncio.run(run_test())


def test_tui_command_history_persists_across_sessions(tmp_path) -> None:
    async def first_session(history_file: Path) -> None:
        app = AgentbahnTui(
            fetch_projects_command=lambda: [],
            fetch_tasks_command=lambda _project_id: [],
            fetch_project_events_command=lambda _project_id: [],
            history_file=history_file,
        )

        async with app.run_test() as pilot:
            await pilot.press(
                "/", "p", "r", "o", "j", "e", "c", "t", " ", "l", "i", "s", "t"
            )
            await pilot.press("enter")
            await pilot.press(
                "/", "t", "a", "s", "k", " ", "l", "i", "s", "t", " ", "7"
            )
            await pilot.press("enter")

    async def second_session(history_file: Path) -> None:
        app = AgentbahnTui(
            fetch_projects_command=lambda: [],
            fetch_tasks_command=lambda _project_id: [],
            fetch_project_events_command=lambda _project_id: [],
            history_file=history_file,
        )

        async with app.run_test() as pilot:
            await pilot.press("up")
            assert app.query_one("#command-input").value == "/task list 7"
            await pilot.press("up")
            assert app.query_one("#command-input").value == "/project list"

    history_file = tmp_path / "command_history"

    asyncio.run(first_session(history_file))
    assert history_file.read_text(encoding="utf-8") == "/project list\n/task list 7\n"
    asyncio.run(second_session(history_file))


def test_tui_non_command_message_streams_to_agent(tmp_path) -> None:
    queries: list[str] = []

    def stream_agent_command(query: str):
        queries.append(query)
        yield CodebaseAgentStreamEvent(type="token", content="Agent ")
        yield CodebaseAgentStreamEvent(type="result", content="done")

    async def run_test() -> None:
        app = AgentbahnTui(
            fetch_projects_command=lambda: [],
            fetch_tasks_command=lambda _project_id: [],
            fetch_project_events_command=lambda _project_id: [],
            stream_agent_command=stream_agent_command,
            history_file=tmp_path / "command_history",
        )

        async with app.run_test() as pilot:
            await pilot.press("B", "u", "i", "l", "d", " ", "i", "t")
            await pilot.press("enter")
            await pilot.pause()
            assert queries == ["Build it"]
            message_output = app.query_one("#message-output")
            assert str(message_output.render()) == "Agent \ndone"

    asyncio.run(run_test())


def test_tui_agent_slash_message_shows_error(tmp_path) -> None:
    queries: list[str] = []

    def stream_agent_command(query: str):
        queries.append(query)
        yield CodebaseAgentStreamEvent(type="result", content="Done")

    async def run_test() -> None:
        app = AgentbahnTui(
            fetch_projects_command=lambda: [],
            fetch_tasks_command=lambda _project_id: [],
            fetch_project_events_command=lambda _project_id: [],
            stream_agent_command=stream_agent_command,
            history_file=tmp_path / "command_history",
        )

        async with app.run_test() as pilot:
            await pilot.press(
                "/", "a", "g", "e", "n", "t", " ", "B", "u", "i", "l", "d"
            )
            await pilot.press("enter")
            await pilot.pause()
            assert queries == []
            message_output = app.query_one("#message-output")
            assert str(message_output.render()) == "Unknown command: /agent Build"

    asyncio.run(run_test())


def test_tui_unknown_slash_message_shows_error(tmp_path) -> None:
    queries: list[str] = []

    def stream_agent_command(query: str):
        queries.append(query)
        yield CodebaseAgentStreamEvent(type="result", content="Done")

    async def run_test() -> None:
        app = AgentbahnTui(
            fetch_projects_command=lambda: [],
            fetch_tasks_command=lambda _project_id: [],
            fetch_project_events_command=lambda _project_id: [],
            stream_agent_command=stream_agent_command,
            history_file=tmp_path / "command_history",
        )

        async with app.run_test() as pilot:
            await pilot.press("/", "u", "n", "k", "n", "o", "w", "n")
            await pilot.press("enter")
            await pilot.pause()
            assert queries == []
            message_output = app.query_one("#message-output")
            assert str(message_output.render()) == "Unknown command: /unknown"

    asyncio.run(run_test())
