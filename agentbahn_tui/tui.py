from __future__ import annotations

from textual.app import App
from textual.app import ComposeResult
from textual.widgets import Static


def get_placeholder_message() -> str:
    return "Hello world from the Agentbahn TUI."


class HelloWorldTui(App[None]):
    """Minimal Textual app used as the package TUI entry point."""

    TITLE = "Agentbahn TUI"

    def compose(self) -> ComposeResult:
        yield Static(get_placeholder_message(), id="hello-world")


def run_tui() -> None:
    HelloWorldTui().run()
