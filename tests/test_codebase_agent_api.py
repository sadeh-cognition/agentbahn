from __future__ import annotations

from ninja.testing import TestClient

from agentbahn.api import api
from agentbahn.codebase_agent.schemas import CodebaseAgentResponse


client = TestClient(api)


def test_codebase_agent_endpoint_returns_matching_codebase_result() -> None:
    response = client.post("/api/codebase-agent", json={"query": "health"})

    assert response.status_code == 200
    payload = CodebaseAgentResponse.model_validate(response.json())
    assert "agentbahn/api.py" in payload.result
    assert "health" in payload.result


def test_codebase_agent_endpoint_rejects_blank_query() -> None:
    response = client.post("/api/codebase-agent", json={"query": "  "})

    assert response.status_code == 422
    assert "Query cannot be blank" in str(response.json())
