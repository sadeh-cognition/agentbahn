from __future__ import annotations

import asyncio
from collections.abc import Sequence

from django.core.management import call_command

from agentbahn.projects.schemas import TaskResponse
from agentbahn_tui import cli
from agentbahn_tui.tui import AgentbahnTui
from agentbahn_tui.tui import CommandHistory
from agentbahn_tui.tui import CommandResult
from agentbahn_tui.tui import format_tasks_output
from agentbahn_tui.tui import get_placeholder_message
from agentbahn_tui.tui import run_tui_command


def test_placeholder_message_is_stable() -> None:
    assert (
        get_placeholder_message()
        == "Enter /project list or /task list PROJECT_ID to fetch data from projectbahn."
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
    output = run_tui_command(
        "/task list 7",
        fetch_projects_command=lambda: [],
        fetch_tasks_command=lambda project_id: [
            TaskResponse(
                id=11,
                project_id=project_id,
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
        ],
    )

    assert output == CommandResult(
        kind="tasks",
        tasks=[
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
        ],
    )


def test_run_tui_command_validates_task_list_arguments() -> None:
    assert (
        run_tui_command(
            "/task list",
            fetch_projects_command=lambda: [],
            fetch_tasks_command=lambda _project_id: [],
        )
        == CommandResult(kind="message", message="Usage: /task list PROJECT_ID")
    )
    assert (
        run_tui_command(
            "/task list abc",
            fetch_projects_command=lambda: [],
            fetch_tasks_command=lambda _project_id: [],
        )
        == CommandResult(kind="message", message="PROJECT_ID must be an integer.")
    )


def test_format_tasks_output_handles_empty_results() -> None:
    assert format_tasks_output([]) == CommandResult(
        kind="message", message="No tasks found."
    )


def test_command_history_navigates_backwards_and_restores_draft() -> None:
    history = CommandHistory(commands=[])

    history.record("/project list")
    history.record("/task list 7")

    assert history.previous("") == "/task list 7"
    assert history.previous("") == "/project list"
    assert history.next() == "/task list 7"
    assert history.next() == ""
    assert history.next() is None


def test_tui_command_history_uses_up_and_down_keys() -> None:
    async def run_test() -> None:
        app = AgentbahnTui(
            fetch_projects_command=lambda: [],
            fetch_tasks_command=lambda _project_id: [],
        )

        async with app.run_test() as pilot:
            await pilot.press("/", "p", "r", "o", "j", "e", "c", "t", " ", "l", "i", "s", "t")
            await pilot.press("enter")
            await pilot.press("/", "t", "a", "s", "k", " ", "l", "i", "s", "t", " ", "7")
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
