from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest


class _FakeBaseLM:
    def __init__(
        self,
        model: str,
        model_type: str = "chat",
        temperature: float | None = 0.0,
        max_tokens: int | None = 1000,
        cache: bool = True,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.kwargs = {"temperature": temperature, "max_tokens": max_tokens, **kwargs}
        self.history: list[dict[str, Any]] = []


@pytest.fixture
def fake_dspy_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("dspy")
    fake_module.BaseLM = _FakeBaseLM
    monkeypatch.setitem(sys.modules, "dspy", fake_module)
    sys.modules.pop("agentbahn.dspy_lm", None)


def test_codex_dspy_lm_builds_responses_payload_and_wraps_response(
    fake_dspy_module: None,
) -> None:
    from agentbahn.dspy_lm import CodexDSPyLM

    class _StubClient:
        def __init__(self) -> None:
            self.payloads: list[dict[str, Any]] = []

        def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
            self.payloads.append(payload)
            return {
                "id": "resp_123",
                "model": "gpt-5.4",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "total_tokens": 14,
                },
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "adapter ok",
                            }
                        ],
                    }
                ],
            }

        def close(self) -> None:
            return None

    client = _StubClient()
    lm = CodexDSPyLM(
        model="gpt-5.4",
        client=client,
        temperature=0.2,
        max_tokens=256,
    )

    response = lm.forward(
        messages=[
            {"role": "system", "content": "Be concise."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Say hello."},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                ],
            },
        ],
        tools=[{"type": "function", "name": "lookup"}],
        response_format={"type": "json_schema", "name": "Answer"},
    )

    assert client.payloads == [
        {
            "model": "gpt-5.4",
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": "Be concise."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Say hello."},
                        {"type": "input_image", "image_url": "https://example.com/cat.png"},
                    ],
                },
            ],
            "max_output_tokens": 256,
            "temperature": 0.2,
            "tools": [{"type": "function", "name": "lookup"}],
            "text": {"format": {"type": "json_schema", "name": "Answer"}},
        }
    ]
    assert response.model == "gpt-5.4"
    assert response.usage.total_tokens == 14
    assert response.output[0].type == "message"
    assert response.output[0].content[0].text == "adapter ok"


def test_codex_dspy_lm_uses_prompt_when_messages_are_omitted(
    fake_dspy_module: None,
) -> None:
    from agentbahn.dspy_lm import CodexDSPyLM

    class _StubClient:
        def __init__(self) -> None:
            self.payloads: list[dict[str, Any]] = []

        def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
            self.payloads.append(payload)
            return {"model": "gpt-5.4", "usage": {}, "output": []}

        def close(self) -> None:
            return None

    client = _StubClient()
    lm = CodexDSPyLM(model="gpt-5.4", client=client)

    lm.forward(prompt="Hello from DSPy")

    assert client.payloads == [
        {
            "model": "gpt-5.4",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello from DSPy"}],
                }
            ],
            "max_output_tokens": 1000,
            "temperature": 0.0,
        }
    ]


def test_codex_dspy_lm_integrates_with_dspy_predict() -> None:
    dspy = pytest.importorskip("dspy")
    import agentbahn.dspy_lm as dspy_lm_module

    importlib.reload(dspy_lm_module)

    class _StubClient:
        def __init__(self) -> None:
            self.payloads: list[dict[str, Any]] = []

        def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
            self.payloads.append(payload)
            return {
                "id": "resp_456",
                "model": "gpt-5.4",
                "usage": {
                    "input_tokens": 21,
                    "output_tokens": 9,
                    "total_tokens": 30,
                },
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "[[ ## answer ## ]]\nRayleigh scattering.",
                            }
                        ],
                    }
                ],
            }

        def close(self) -> None:
            return None

    class ExplainSky(dspy.Signature):
        question = dspy.InputField()
        answer = dspy.OutputField()

    client = _StubClient()
    lm = dspy_lm_module.CodexDSPyLM(model="gpt-5.4", client=client)
    predictor = dspy.Predict(ExplainSky)

    with dspy.context(lm=lm):
        result = predictor(question="Why is the sky blue?")

    assert result.answer == "Rayleigh scattering."
    assert len(client.payloads) == 1
    assert client.payloads[0]["model"] == "gpt-5.4"
    assert client.payloads[0]["input"][0]["role"] == "system"
    assert client.payloads[0]["input"][1]["role"] == "user"
    assert "question" in client.payloads[0]["input"][1]["content"][0]["text"]
    assert "Why is the sky blue?" in client.payloads[0]["input"][1]["content"][0]["text"]
