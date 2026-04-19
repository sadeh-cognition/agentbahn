from __future__ import annotations

from collections.abc import Sequence

from django.core.management import call_command

from agentbahn_tui import cli
from agentbahn_tui.tui import get_placeholder_message


def test_placeholder_message_is_stable() -> None:
    assert get_placeholder_message() == "Hello world from the Agentbahn TUI."


def test_management_command_starts_tui(monkeypatch) -> None:
    calls: list[str] = []

    def fake_run_tui() -> None:
        calls.append("started")

    monkeypatch.setattr("agentbahn_tui.management.commands.start_tui.run_tui", fake_run_tui)

    call_command("start_tui")

    assert calls == ["started"]


def test_console_script_dispatches_to_management_command(monkeypatch) -> None:
    executed_argv: list[str] = []

    def fake_execute_from_command_line(argv: Sequence[str]) -> None:
        executed_argv.extend(argv)

    monkeypatch.delenv("DJANGO_SETTINGS_MODULE", raising=False)
    monkeypatch.setattr(cli, "execute_from_command_line", fake_execute_from_command_line)

    exit_code = cli.main(["--help"])

    assert exit_code == 0
    assert executed_argv == ["agentbahn-tui", "start_tui", "--help"]
    assert cli.os.environ["DJANGO_SETTINGS_MODULE"] == "agentbahn.settings"
