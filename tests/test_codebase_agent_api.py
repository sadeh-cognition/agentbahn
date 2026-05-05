from __future__ import annotations

import asyncio
import json
from typing import Any

from asgiref.sync import async_to_sync
from django.test import Client
from model_bakery import baker
from ninja.testing import TestAsyncClient

from agentbahn.api import api
from agentbahn.codebase_agent.agent import DefaultAgent
from agentbahn.codebase_agent.services import _build_codebase_agent_async
from agentbahn.llms.models import encrypt_api_key
from agentbahn.llms.models import LlmConfiguration


client = Client()
ninja_client = TestAsyncClient(api)


def post_json(path: str, payload: dict[str, Any]) -> Any:
    return client.post(
        path,
        data=json.dumps(payload),
        content_type="application/json",
    )


def test_codebase_agent_endpoint_rejects_blank_query() -> None:
    response = post_json("/api/codebase-agent", {"query": "  "})

    assert response.status_code == 422
    assert "Query cannot be blank" in str(response.json())


def test_codebase_agent_endpoint_rejects_missing_llm_configuration(db) -> None:
    response = asyncio.run(
        ninja_client.post(
            "/api/codebase-agent",
            json={"query": "Build feature", "llm_config_id": 99},
        )
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "LLM configuration 99 was not found."


def test_codebase_agent_builds_llm_configuration_outside_async_context(db) -> None:
    baker.make(
        LlmConfiguration,
        pk=1,
        name="Groq fast",
        provider="groq",
        llm_name="llama-3.1-8b-instant",
        encrypted_api_key=encrypt_api_key("secret-key"),
    )

    agent = async_to_sync(_build_codebase_agent_async)()

    assert isinstance(agent, DefaultAgent)
    assert agent.lm.model == "groq/llama-3.1-8b-instant"


def test_codebase_agent_builds_selected_llm_configuration(db) -> None:
    baker.make(
        LlmConfiguration,
        pk=1,
        name="Groq fast",
        provider="groq",
        llm_name="llama-3.1-8b-instant",
        encrypted_api_key=encrypt_api_key("secret-key"),
    )
    baker.make(
        LlmConfiguration,
        pk=2,
        name="OpenAI main",
        provider="openai",
        llm_name="gpt-5.5",
        encrypted_api_key=encrypt_api_key("other-key"),
    )

    agent = async_to_sync(_build_codebase_agent_async)(llm_config_id=2)

    assert isinstance(agent, DefaultAgent)
    assert agent.lm.model == "openai/gpt-5.5"


def test_codebase_agent_rejects_missing_selected_llm_configuration(db) -> None:
    try:
        async_to_sync(_build_codebase_agent_async)(llm_config_id=99)
    except ValueError as exc:
        assert str(exc) == "LLM configuration 99 was not found."
    else:
        raise AssertionError("Expected missing LLM configuration to raise ValueError.")
