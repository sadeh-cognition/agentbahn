from __future__ import annotations

from collections.abc import Callable

import httpx

from agentbahn_tui.llms import get_agentbahn_api_base_url


class BackendUnavailableError(RuntimeError):
    """Raised when the agentbahn backend cannot be reached."""


def check_backend_server_running(
    *,
    transport: httpx.BaseTransport | None = None,
    client_factory: Callable[..., httpx.Client] = httpx.Client,
) -> None:
    base_url = get_agentbahn_api_base_url()
    try:
        with client_factory(
            base_url=base_url, transport=transport, timeout=2.0
        ) as client:
            response = client.get("/api/health")
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise BackendUnavailableError(
            f"agentbahn backend server is not running or is unreachable at {base_url}."
        ) from exc
