from __future__ import annotations

import base64
import json
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from agentbahn.codex_openai_client import CodexOpenAIClient
from agentbahn.codex_openai_client import DEFAULT_CHATGPT_BASE_URL
from agentbahn.codex_openai_client import DEFAULT_OPENAI_BASE_URL
from agentbahn.codex_openai_client import resolve_codex_credentials


def _jwt(payload: dict[str, Any]) -> str:
    header = {"alg": "none", "typ": "JWT"}
    parts = []
    for value in (header, payload, "signature"):
        raw = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
        encoded = base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")
        parts.append(encoded)
    return ".".join(parts)


def _chatgpt_token(*, account_id: str, fedramp: bool = False) -> str:
    return _jwt(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": account_id,
                "chatgpt_account_is_fedramp": fedramp,
            }
        }
    )


def _write_auth_json(
    codex_home: Path,
    *,
    auth_mode: str = "chatgpt",
    access_token: str | None = None,
    refresh_token: str | None = None,
    id_token: str | None = None,
    account_id: str | None = None,
    last_refresh: datetime | None = None,
    api_key: str | None = None,
) -> None:
    payload: dict[str, Any] = {"auth_mode": auth_mode, "OPENAI_API_KEY": api_key}
    if auth_mode == "chatgpt":
        payload["tokens"] = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "id_token": id_token,
            "account_id": account_id,
        }
        payload["last_refresh"] = (
            (last_refresh or datetime.now(tz=UTC)).isoformat().replace("+00:00", "Z")
        )
    auth_file = codex_home / "auth.json"
    auth_file.parent.mkdir(parents=True, exist_ok=True)
    auth_file.write_text(json.dumps(payload, indent=2))


@pytest.fixture
def clear_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CODEX_API_KEY",
        "CODEX_HOME",
        "CODEX_REFRESH_TOKEN_URL_OVERRIDE",
        "OPENAI_ORGANIZATION",
        "OPENAI_PROJECT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_resolve_chatgpt_credentials_from_codex_auth_file(tmp_path: Path) -> None:
    account_id = "workspace-123"
    token = _chatgpt_token(account_id=account_id, fedramp=True)
    _write_auth_json(
        tmp_path,
        access_token=token,
        refresh_token="refresh-token",
        id_token=token,
        account_id=account_id,
    )

    credentials = resolve_codex_credentials(codex_home=tmp_path)

    assert credentials.auth_mode == "chatgpt"
    assert credentials.token == token
    assert credentials.account_id == account_id
    assert credentials.base_url == DEFAULT_CHATGPT_BASE_URL
    assert credentials.is_fedramp_account is True


def test_codex_api_key_env_takes_precedence_over_auth_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = _chatgpt_token(account_id="workspace-123")
    _write_auth_json(
        tmp_path,
        access_token=token,
        refresh_token="refresh-token",
        id_token=token,
        account_id="workspace-123",
    )
    monkeypatch.setenv("CODEX_API_KEY", "sk-from-env")

    credentials = resolve_codex_credentials(codex_home=tmp_path)

    assert credentials.auth_mode == "api_key"
    assert credentials.token == "sk-from-env"
    assert credentials.account_id is None
    assert credentials.base_url == DEFAULT_OPENAI_BASE_URL


class _TestServer:
    def __init__(self) -> None:
        self.response_auth_headers: list[str | None] = []
        self.chatgpt_account_headers: list[str | None] = []
        self.refresh_payloads: list[dict[str, Any]] = []
        self.sse_requests = 0
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length)
                body = json.loads(raw_body or b"{}")

                if self.path == "/backend-api/codex/responses":
                    auth_header = self.headers.get("Authorization")
                    outer.response_auth_headers.append(auth_header)
                    outer.chatgpt_account_headers.append(self.headers.get("ChatGPT-Account-ID"))

                    if auth_header == "Bearer stale-access":
                        self.send_response(401)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(b'{"error":"unauthorized"}')
                        return

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"ok": True, "echo": body}).encode())
                    return

                if self.path == "/oauth/token":
                    outer.refresh_payloads.append(body)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    refreshed_token = _chatgpt_token(account_id="workspace-123")
                    self.wfile.write(
                        json.dumps(
                            {
                                "access_token": "fresh-access",
                                "refresh_token": "fresh-refresh",
                                "id_token": refreshed_token,
                            }
                        ).encode()
                    )
                    return

                if self.path == "/v1/responses":
                    outer.sse_requests += 1
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    chunks = [
                        'event: response.created\ndata: {"type":"response.created","response":{"id":"resp-1"}}\n\n',
                        'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"print(1)"}\n\n',
                        'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp-1"}}\n\n',
                    ]
                    for chunk in chunks:
                        self.wfile.write(chunk.encode())
                        self.wfile.flush()
                    return

                self.send_response(404)
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                return

        return Handler


@pytest.fixture
def test_server() -> Iterator[_TestServer]:
    server = _TestServer()
    server.start()
    try:
        yield server
    finally:
        server.close()


def test_client_refreshes_chatgpt_token_on_401_and_retries_request(
    tmp_path: Path,
    test_server: _TestServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_token = _chatgpt_token(account_id="workspace-123")
    _write_auth_json(
        tmp_path,
        access_token="stale-access",
        refresh_token="stale-refresh",
        id_token=stale_token,
        account_id="workspace-123",
        last_refresh=datetime.now(tz=UTC),
    )
    monkeypatch.setenv(
        "CODEX_REFRESH_TOKEN_URL_OVERRIDE",
        f"{test_server.base_url}/oauth/token",
    )

    with CodexOpenAIClient(codex_home=tmp_path, base_url=f"{test_server.base_url}/backend-api/codex") as client:
        result = client.create_response({"model": "gpt-5.4", "input": [], "stream": False})

    assert result["ok"] is True
    assert test_server.response_auth_headers == ["Bearer stale-access", "Bearer fresh-access"]
    assert test_server.chatgpt_account_headers == ["workspace-123", "workspace-123"]
    assert test_server.refresh_payloads == [
        {
            "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
            "grant_type": "refresh_token",
            "refresh_token": "stale-refresh",
        }
    ]

    auth_json = json.loads((tmp_path / "auth.json").read_text())
    assert auth_json["tokens"]["access_token"] == "fresh-access"
    assert auth_json["tokens"]["refresh_token"] == "fresh-refresh"


def test_stream_response_parses_sse_events(
    monkeypatch: pytest.MonkeyPatch,
    test_server: _TestServer,
) -> None:
    monkeypatch.setenv("CODEX_API_KEY", "sk-test")

    with CodexOpenAIClient(base_url=f"{test_server.base_url}/v1") as client:
        events = list(client.stream_response({"model": "gpt-5.4", "input": [], "stream": True}))

    assert test_server.sse_requests == 1
    assert [event.type for event in events] == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]
    assert events[1].json_data == {
        "type": "response.output_text.delta",
        "delta": "print(1)",
    }
