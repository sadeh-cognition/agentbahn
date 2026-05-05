from __future__ import annotations

from typing import Any

from agentbahn.llms.openai_lm import DEFAULT_FLEX_TIMEOUT_SECONDS
from agentbahn.llms.openai_lm import OpenAIFlexLM


class FakeResponses:
    def __init__(self) -> None:
        self.request: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> object:
        self.request = kwargs
        return object()


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()
        self.timeout: float | None = None

    def with_options(self, *, timeout: float) -> FakeClient:
        self.timeout = timeout
        return self


def test_openai_flex_lm_sends_service_tier_flex_and_long_timeout() -> None:
    client = FakeClient()
    lm = OpenAIFlexLM(
        model="gpt-5.5",
        api_key="secret-key",
        client=client,
        temperature=0.2,
        max_tokens=512,
    )

    lm.forward(prompt="Summarize this.")

    assert client.timeout == DEFAULT_FLEX_TIMEOUT_SECONDS
    assert client.responses.request == {
        "model": "gpt-5.5",
        "input": "Summarize this.",
        "temperature": 0.2,
        "max_output_tokens": 512,
        "service_tier": "flex",
    }


def test_openai_flex_lm_preserves_messages_and_allows_per_call_options() -> None:
    client = FakeClient()
    lm = OpenAIFlexLM(model="gpt-5.5", api_key="secret-key", client=client)

    lm.forward(
        messages=[
            {"role": "developer", "content": "Be concise."},
            {"role": "user", "content": "Explain flex processing."},
        ],
        temperature=0.7,
        rollout_id=None,
    )

    assert client.responses.request == {
        "model": "gpt-5.5",
        "input": [
            {"role": "developer", "content": "Be concise."},
            {"role": "user", "content": "Explain flex processing."},
        ],
        "temperature": 0.7,
        "max_output_tokens": 1000,
        "service_tier": "flex",
    }
