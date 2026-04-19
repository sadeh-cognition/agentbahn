from __future__ import annotations

import djclick as click

from agentbahn_tui.tui import run_tui


@click.command()
def start_tui() -> None:
    """Start the Textual TUI."""
    run_tui()
