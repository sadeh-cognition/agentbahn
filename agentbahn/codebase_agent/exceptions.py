class InterruptAgentFlow(Exception):
    """Raised to interrupt the agent flow and add messages."""

    def __init__(self, *messages: dict):
        self.messages = messages
        super().__init__()


class LimitsExceeded(InterruptAgentFlow):
    """Raised when the agent has exceeded its cost or step limit."""


class Submitted(InterruptAgentFlow):
    """Raised when the agent has completed its task."""
