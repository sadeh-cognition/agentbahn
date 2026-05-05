from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator

import httpx
from pydantic import TypeAdapter

from agentbahn.codebase_agent.schemas import CodebaseAgentRequest
from agentbahn.codebase_agent.schemas import CodebaseAgentStreamEvent
from agentbahn_tui.llms import get_agentbahn_api_base_url


def stream_codebase_agent(
    query: str,
    llm_config_id: int | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    client_factory: Callable[..., httpx.Client] = httpx.Client,
) -> Iterator[CodebaseAgentStreamEvent]:
    payload = CodebaseAgentRequest(query=query, llm_config_id=llm_config_id)
    event_adapter = TypeAdapter(CodebaseAgentStreamEvent)

    with client_factory(
        base_url=get_agentbahn_api_base_url(),
        transport=transport,
        timeout=httpx.Timeout(5.0, read=None),
    ) as client:
        with client.stream(
            "POST",
            "/api/codebase-agent",
            json=payload.model_dump(
                mode="json",
                exclude_defaults=True,
                exclude_none=True,
            ),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                yield event_adapter.validate_json(line)
