from __future__ import annotations

from collections.abc import Callable

import httpx
from django.conf import settings
from pydantic import TypeAdapter

from agentbahn.projects.schemas import ProjectListResponse


def get_api_base_url() -> str:
    return str(settings.PROJECTBAHN_API_BASE_URL).rstrip("/")


def fetch_projects(
    *,
    transport: httpx.BaseTransport | None = None,
    client_factory: Callable[..., httpx.Client] = httpx.Client,
) -> ProjectListResponse:
    with client_factory(
        base_url=get_api_base_url(), transport=transport, timeout=5.0
    ) as client:
        response = client.get("/api/projects")
        response.raise_for_status()
    return TypeAdapter(ProjectListResponse).validate_python(response.json())
