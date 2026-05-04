from __future__ import annotations

from collections.abc import Callable

import httpx
from django.conf import settings
from pydantic import TypeAdapter

from agentbahn.llms.schemas import LlmConfigListResponse
from agentbahn.llms.schemas import LlmConfigLookupResponse
from agentbahn.llms.schemas import LlmConfigResponse
from agentbahn.llms.schemas import LlmConfigUpsertRequest


def get_agentbahn_api_base_url() -> str:
    return str(settings.API_BASE_URL).rstrip("/")


def fetch_llm_config(
    *,
    transport: httpx.BaseTransport | None = None,
    client_factory: Callable[..., httpx.Client] = httpx.Client,
) -> LlmConfigLookupResponse:
    with client_factory(
        base_url=get_agentbahn_api_base_url(),
        transport=transport,
        timeout=5.0,
    ) as client:
        response = client.get("/api/llm-config")
        response.raise_for_status()
    return TypeAdapter(LlmConfigLookupResponse).validate_python(response.json())


def fetch_llm_configs(
    *,
    transport: httpx.BaseTransport | None = None,
    client_factory: Callable[..., httpx.Client] = httpx.Client,
) -> LlmConfigListResponse:
    with client_factory(
        base_url=get_agentbahn_api_base_url(),
        transport=transport,
        timeout=5.0,
    ) as client:
        response = client.get("/api/llm-configs")
        response.raise_for_status()
    return TypeAdapter(LlmConfigListResponse).validate_python(response.json())


def save_llm_config(
    payload: LlmConfigUpsertRequest,
    *,
    transport: httpx.BaseTransport | None = None,
    client_factory: Callable[..., httpx.Client] = httpx.Client,
) -> LlmConfigResponse:
    with client_factory(
        base_url=get_agentbahn_api_base_url(),
        transport=transport,
        timeout=5.0,
    ) as client:
        response = client.post(
            "/api/llm-config",
            json=payload.model_dump(
                mode="json",
                exclude_defaults=True,
                exclude_none=True,
            ),
        )
        response.raise_for_status()
    return TypeAdapter(LlmConfigResponse).validate_python(response.json())
