from __future__ import annotations

import json

import httpx
from django.conf import settings

from agentbahn_tui.agents import stream_codebase_agent


def test_stream_codebase_agent_posts_request_and_parses_ndjson_events() -> None:
    settings.API_BASE_URL = "http://testserver"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            content=(
                json.dumps({"type": "token", "content": "Hel"}) + "\n"
                "\n"
                + json.dumps({"type": "token", "content": "lo"})
                + "\n"
                + json.dumps({"type": "result", "content": "Done"})
                + "\n"
            ),
        )

    events = list(
        stream_codebase_agent(
            "  Build feature  ",
            llm_config_id=7,
            transport=httpx.MockTransport(handler),
        )
    )

    assert [event.type for event in events] == ["token", "token", "result"]
    assert [event.content for event in events] == ["Hel", "lo", "Done"]
    assert len(requests) == 1
    assert requests[0].url == "http://testserver/api/codebase-agent"
    assert json.loads(requests[0].content) == {
        "query": "Build feature",
        "llm_config_id": 7,
    }
