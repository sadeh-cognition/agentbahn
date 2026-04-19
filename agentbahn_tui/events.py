from __future__ import annotations

from collections.abc import Callable

import httpx
from pydantic import TypeAdapter

from agentbahn.projects.schemas import EventLogListResponse
from agentbahn.projects.schemas import EventLogPageResponse
from agentbahn_tui.projects import get_api_base_url


def fetch_event_logs(
    entity_type: str,
    entity_id: int,
    *,
    transport: httpx.BaseTransport | None = None,
    client_factory: Callable[..., httpx.Client] = httpx.Client,
    page_size: int = 200,
) -> EventLogListResponse:
    collected_events: EventLogListResponse = []

    with client_factory(
        base_url=get_api_base_url(), transport=transport, timeout=5.0
    ) as client:
        page = 1
        while True:
            response = client.get(
                "/api/event-logs",
                params={
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "page": page,
                    "page_size": page_size,
                },
            )
            response.raise_for_status()

            event_page = TypeAdapter(EventLogPageResponse).validate_python(
                response.json()
            )
            collected_events.extend(event_page.items)
            if page * event_page.page_size >= event_page.total:
                break
            page += 1

    return collected_events
