from __future__ import annotations

from collections.abc import Callable

import httpx
from pydantic import TypeAdapter

from agentbahn.projects.schemas import TaskListResponse
from agentbahn_tui.projects import get_api_base_url


def fetch_tasks(
    project_id: int,
    *,
    transport: httpx.BaseTransport | None = None,
    client_factory: Callable[..., httpx.Client] = httpx.Client,
) -> TaskListResponse:
    with client_factory(base_url=get_api_base_url(), transport=transport, timeout=5.0) as client:
        response = client.get("/api/tasks", params={"project_id": project_id})
        response.raise_for_status()
    return TypeAdapter(TaskListResponse).validate_python(response.json())
