from __future__ import annotations

from collections.abc import Callable

import httpx
from pydantic import TypeAdapter

from agentbahn.projects.schemas import FeatureListResponse
from agentbahn_tui.projects import get_api_base_url


def fetch_features(
    project_id: int | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    client_factory: Callable[..., httpx.Client] = httpx.Client,
) -> FeatureListResponse:
    with client_factory(
        base_url=get_api_base_url(), transport=transport, timeout=5.0
    ) as client:
        response = client.get("/api/features")
        response.raise_for_status()

    features = TypeAdapter(FeatureListResponse).validate_python(response.json())
    if project_id is None:
        return features
    return [feature for feature in features if feature.project_id == project_id]
