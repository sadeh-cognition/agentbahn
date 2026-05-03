from __future__ import annotations

import json
from typing import Any

from asgiref.sync import async_to_sync
from django.test import Client
from model_bakery import baker

from agentbahn.codebase_agent.agent import DefaultAgent
from agentbahn.codebase_agent.services import _build_codebase_agent_async
from agentbahn.llms.models import encrypt_api_key
from agentbahn.llms.models import LlmConfiguration


client = Client()


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


def test_codebase_agent_builds_llm_configuration_outside_async_context(db) -> None:
    baker.make(
        LlmConfiguration,
        pk=1,
        provider="groq",
        llm_name="llama-3.1-8b-instant",
        encrypted_api_key=encrypt_api_key("secret-key"),
    )

    agent = async_to_sync(_build_codebase_agent_async)()

    assert isinstance(agent, DefaultAgent)
    assert agent.lm.model == "groq/llama-3.1-8b-instant"
