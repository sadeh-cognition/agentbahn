from __future__ import annotations

from typing import Any

import djclick as click
import dspy
from dspy.utils.callback import BaseCallback

from agentbahn.llms.models import decrypt_api_key
from agentbahn.llms.openai_lm import OpenAIFlexLM
from agentbahn.llms.services import get_llm_configuration


class WeatherAnswer(dspy.Signature):
    """Answer the user's weather question using the provided tools."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class ConsoleLoggingCallback(BaseCallback):
    """Print selected DSPy callback events to the command console."""

    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        click.echo(
            f"[lm:start] {call_id} {instance.__class__.__name__} "
            f"inputs={self._format(inputs)}"
        )

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        click.echo(
            f"[lm:end] {call_id} outputs={self._format(outputs)} "
            f"exception={self._format(exception)}"
        )

    def on_tool_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        tool_name = getattr(instance, "name", instance.__class__.__name__)
        click.echo(f"[tool:start] {call_id} {tool_name} inputs={self._format(inputs)}")

    def on_tool_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        click.echo(
            f"[tool:end] {call_id} outputs={self._format(outputs)} "
            f"exception={self._format(exception)}"
        )

    @staticmethod
    def _format(value: Any) -> str:
        text = repr(value)
        if len(text) > 500:
            return f"{text[:497]}..."
        return text


def get_weather(city: str) -> str:
    """Return fake weather for a city."""
    return f"The weather in {city} is sunny and 72 degrees."


def build_openai_flex_lm() -> OpenAIFlexLM:
    config = get_llm_configuration()
    if config is None:
        raise click.ClickException("No LLM configuration found.")
    if config.provider.strip().lower() != "openai":
        raise click.ClickException(
            "The configured LLM provider must be 'openai' to use OpenAIFlexLM."
        )

    return OpenAIFlexLM(
        model=config.llm_name,
        api_key=decrypt_api_key(config.encrypted_api_key),
    )


@click.command()
def command() -> None:
    """Run a small DSPy ReAct example with OpenAI Flex and console logging."""
    lm = build_openai_flex_lm()
    react_agent = dspy.ReAct(WeatherAnswer, tools=[get_weather], max_iters=3)

    with dspy.context(lm=lm, callbacks=[ConsoleLoggingCallback()]):
        prediction = react_agent(question="What is the weather in Berlin?")

    click.secho("\nPrediction", fg="green")
    click.echo(prediction)
