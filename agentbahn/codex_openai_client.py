from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import httpx

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_CHATGPT_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_ORIGINATOR = "codex_cli_rs"
DEFAULT_VERSION_HEADER = "python-port"
TOKEN_REFRESH_INTERVAL_DAYS = 8
REFRESH_TOKEN_URL = "https://auth.openai.com/oauth/token"
REFRESH_TOKEN_URL_OVERRIDE_ENV_VAR = "CODEX_REFRESH_TOKEN_URL_OVERRIDE"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"


class CodexAuthError(RuntimeError):
    """Raised when Codex-compatible credentials cannot be loaded or refreshed."""


class CodexResponsesError(RuntimeError):
    """Raised when the responses endpoint returns an error."""


def _trimmed_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _isoformat_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3 or not all(parts):
        raise CodexAuthError("invalid JWT format in Codex auth data")
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    decoded = base64.urlsafe_b64decode(payload + padding)
    value = json.loads(decoded)
    if not isinstance(value, dict):
        raise CodexAuthError("invalid JWT payload in Codex auth data")
    return value


def _extract_chatgpt_auth_claims(token: str) -> dict[str, Any]:
    payload = _decode_jwt_payload(token)
    claims = payload.get("https://api.openai.com/auth")
    if isinstance(claims, dict):
        return claims
    return {}


def _extract_account_id(*tokens: str | None) -> str | None:
    for token in tokens:
        if not token:
            continue
        try:
            claims = _extract_chatgpt_auth_claims(token)
        except CodexAuthError:
            continue
        account_id = claims.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    return None


def _is_fedramp_account(*tokens: str | None) -> bool:
    for token in tokens:
        if not token:
            continue
        try:
            claims = _extract_chatgpt_auth_claims(token)
        except CodexAuthError:
            continue
        if bool(claims.get("chatgpt_account_is_fedramp")):
            return True
    return False


def find_codex_home() -> Path:
    codex_home = _trimmed_env("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser()
    return Path.home() / ".codex"


def _auth_file_for(codex_home: Path) -> Path:
    return codex_home / "auth.json"


def _load_auth_json(auth_file: Path) -> dict[str, Any]:
    if not auth_file.exists():
        raise CodexAuthError(f"missing Codex auth file: {auth_file}")
    data = json.loads(auth_file.read_text())
    if not isinstance(data, dict):
        raise CodexAuthError(f"invalid Codex auth file: {auth_file}")
    return data


@dataclass(slots=True)
class CodexCredentials:
    auth_mode: Literal["api_key", "chatgpt"]
    token: str
    account_id: str | None = None
    refresh_token: str | None = None
    id_token: str | None = None
    is_fedramp_account: bool = False
    base_url: str = DEFAULT_OPENAI_BASE_URL
    last_refresh: datetime | None = None
    auth_file: Path | None = None
    auth_json: dict[str, Any] | None = None

    @property
    def is_chatgpt(self) -> bool:
        return self.auth_mode == "chatgpt"

    def should_refresh(self) -> bool:
        if not self.is_chatgpt or not self.refresh_token or self.last_refresh is None:
            return False
        return self.last_refresh < datetime.now(tz=UTC) - timedelta(days=TOKEN_REFRESH_INTERVAL_DAYS)

    def apply_headers(self, headers: dict[str, str]) -> None:
        headers["Authorization"] = f"Bearer {self.token}"
        if self.account_id:
            headers["ChatGPT-Account-ID"] = self.account_id
        if self.is_fedramp_account:
            headers["X-OpenAI-Fedramp"] = "true"


def resolve_codex_credentials(
    *,
    codex_home: Path | None = None,
    enable_codex_api_key_env: bool = True,
    base_url: str | None = None,
) -> CodexCredentials:
    if enable_codex_api_key_env:
        api_key = _trimmed_env("CODEX_API_KEY")
        if api_key:
            return CodexCredentials(
                auth_mode="api_key",
                token=api_key,
                base_url=base_url or DEFAULT_OPENAI_BASE_URL,
            )

    auth_file = _auth_file_for(codex_home or find_codex_home())
    auth_json = _load_auth_json(auth_file)
    auth_mode = auth_json.get("auth_mode")

    if auth_mode == "api_key" or (auth_mode is None and auth_json.get("OPENAI_API_KEY")):
        api_key = auth_json.get("OPENAI_API_KEY")
        if not isinstance(api_key, str) or not api_key.strip():
            raise CodexAuthError("Codex auth is configured for API-key mode but no key was stored")
        return CodexCredentials(
            auth_mode="api_key",
            token=api_key.strip(),
            base_url=base_url or DEFAULT_OPENAI_BASE_URL,
            auth_file=auth_file,
            auth_json=auth_json,
        )

    tokens = auth_json.get("tokens")
    if not isinstance(tokens, dict):
        raise CodexAuthError("Codex auth is configured for ChatGPT mode but tokens are missing")

    access_token = tokens.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise CodexAuthError("Codex ChatGPT auth is missing an access token")

    id_token = tokens.get("id_token")
    refresh_token = tokens.get("refresh_token")
    account_id = tokens.get("account_id")
    if not isinstance(account_id, str) or not account_id:
        account_id = _extract_account_id(
            id_token if isinstance(id_token, str) else None,
            access_token,
        )

    return CodexCredentials(
        auth_mode="chatgpt",
        token=access_token.strip(),
        account_id=account_id,
        refresh_token=refresh_token.strip() if isinstance(refresh_token, str) and refresh_token.strip() else None,
        id_token=id_token.strip() if isinstance(id_token, str) and id_token.strip() else None,
        is_fedramp_account=_is_fedramp_account(
            id_token if isinstance(id_token, str) else None,
            access_token,
        ),
        base_url=base_url or DEFAULT_CHATGPT_BASE_URL,
        last_refresh=_parse_timestamp(auth_json.get("last_refresh")),
        auth_file=auth_file,
        auth_json=auth_json,
    )


@dataclass(slots=True)
class SseEvent:
    event: str | None
    data: str
    json_data: dict[str, Any] | None

    @property
    def type(self) -> str | None:
        if self.json_data is None:
            return None
        event_type = self.json_data.get("type")
        return event_type if isinstance(event_type, str) else None


class CodexOpenAIClient:
    """Low-level Responses API client that mirrors Codex CLI auth behavior."""

    def __init__(
        self,
        *,
        credentials: CodexCredentials | None = None,
        codex_home: Path | None = None,
        enable_codex_api_key_env: bool = True,
        base_url: str | None = None,
        timeout: float = 120.0,
        version_header: str = DEFAULT_VERSION_HEADER,
    ) -> None:
        self.credentials = credentials or resolve_codex_credentials(
            codex_home=codex_home,
            enable_codex_api_key_env=enable_codex_api_key_env,
            base_url=base_url,
        )
        self.base_url = base_url or self.credentials.base_url
        self.version_header = version_header
        self._http = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> CodexOpenAIClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def default_headers(self, *, accept_sse: bool = False) -> dict[str, str]:
        headers = {
            "originator": DEFAULT_ORIGINATOR,
            "version": self.version_header,
            "User-Agent": f"{DEFAULT_ORIGINATOR}/{self.version_header}",
        }
        organization = _trimmed_env("OPENAI_ORGANIZATION")
        if organization:
            headers["OpenAI-Organization"] = organization
        project = _trimmed_env("OPENAI_PROJECT")
        if project:
            headers["OpenAI-Project"] = project
        if accept_sse:
            headers["Accept"] = "text/event-stream"
        self.credentials.apply_headers(headers)
        return headers

    def url_for_path(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def build_text_message_input(self, text: str) -> list[dict[str, Any]]:
        return [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            }
        ]

    def build_code_request(
        self,
        *,
        prompt: str,
        model: str,
        instructions: str = "",
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        store: bool = True,
        include: list[str] | None = None,
        reasoning: dict[str, Any] | None = None,
        service_tier: str | None = None,
        prompt_cache_key: str | None = None,
        text: dict[str, Any] | None = None,
        client_metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return {
            "model": model,
            "instructions": instructions,
            "input": self.build_text_message_input(prompt),
            "tools": tools or [],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "reasoning": reasoning,
            "store": store,
            "stream": stream,
            "include": include or [],
            "service_tier": service_tier,
            "prompt_cache_key": prompt_cache_key,
            "text": text,
            "client_metadata": client_metadata,
        }

    def create_response(self, payload: Mapping[str, Any], *, path: str = "responses") -> dict[str, Any]:
        response = self._send_json_request(path=path, payload=payload)
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise CodexResponsesError("responses endpoint returned non-JSON content") from exc

    def stream_response(
        self,
        payload: Mapping[str, Any],
        *,
        path: str = "responses",
    ) -> Iterator[SseEvent]:
        response = self._send_stream_request(path=path, payload=payload)
        try:
            yield from self._iter_sse_events(response)
        finally:
            response.close()

    def _send_json_request(self, *, path: str, payload: Mapping[str, Any]) -> httpx.Response:
        self._maybe_refresh_before_request()
        refreshed_after_401 = False
        for attempt in range(4):
            try:
                response = self._http.post(
                    self.url_for_path(path),
                    headers=self.default_headers(),
                    json=dict(payload),
                )
            except httpx.HTTPError as exc:
                if attempt == 3:
                    raise CodexResponsesError(f"request failed: {exc}") from exc
                time.sleep(0.2 * (2**attempt))
                continue

            if response.status_code == 401 and self.credentials.is_chatgpt and not refreshed_after_401:
                response.close()
                self._refresh_chatgpt_tokens()
                refreshed_after_401 = True
                continue

            if response.status_code >= 500 and attempt < 3:
                response.close()
                time.sleep(0.2 * (2**attempt))
                continue

            self._raise_for_status(response)
            return response

        raise CodexResponsesError("request failed after retry exhaustion")

    def _send_stream_request(self, *, path: str, payload: Mapping[str, Any]) -> httpx.Response:
        self._maybe_refresh_before_request()
        refreshed_after_401 = False
        for attempt in range(4):
            request = self._http.build_request(
                "POST",
                self.url_for_path(path),
                headers=self.default_headers(accept_sse=True),
                json=dict(payload),
            )
            try:
                response = self._http.send(request, stream=True)
            except httpx.HTTPError as exc:
                if attempt == 3:
                    raise CodexResponsesError(f"stream request failed: {exc}") from exc
                time.sleep(0.2 * (2**attempt))
                continue

            if response.status_code == 401 and self.credentials.is_chatgpt and not refreshed_after_401:
                response.close()
                self._refresh_chatgpt_tokens()
                refreshed_after_401 = True
                continue

            if response.status_code >= 500 and attempt < 3:
                response.close()
                time.sleep(0.2 * (2**attempt))
                continue

            self._raise_for_status(response)
            return response

        raise CodexResponsesError("stream request failed after retry exhaustion")

    def _maybe_refresh_before_request(self) -> None:
        if self.credentials.should_refresh():
            self._refresh_chatgpt_tokens()

    def _refresh_chatgpt_tokens(self) -> None:
        if not self.credentials.is_chatgpt or not self.credentials.refresh_token:
            raise CodexAuthError("ChatGPT refresh requested but no refresh token is available")

        refresh_url = _trimmed_env(REFRESH_TOKEN_URL_OVERRIDE_ENV_VAR) or REFRESH_TOKEN_URL
        response = self._http.post(
            refresh_url,
            headers={
                "originator": DEFAULT_ORIGINATOR,
                "User-Agent": f"{DEFAULT_ORIGINATOR}/{self.version_header}",
                "Content-Type": "application/json",
            },
            json={
                "client_id": CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": self.credentials.refresh_token,
            },
        )
        self._raise_for_status(response, prefix="token refresh failed")

        data = response.json()
        if not isinstance(data, dict):
            raise CodexAuthError("token refresh returned an invalid JSON payload")

        access_token = data.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            raise CodexAuthError("token refresh did not return an access token")

        new_id_token = data.get("id_token")
        if isinstance(new_id_token, str) and new_id_token.strip():
            self.credentials.id_token = new_id_token.strip()
        new_refresh_token = data.get("refresh_token")
        if isinstance(new_refresh_token, str) and new_refresh_token.strip():
            self.credentials.refresh_token = new_refresh_token.strip()

        self.credentials.token = access_token.strip()
        self.credentials.account_id = self.credentials.account_id or _extract_account_id(
            self.credentials.id_token,
            self.credentials.token,
        )
        self.credentials.is_fedramp_account = _is_fedramp_account(
            self.credentials.id_token,
            self.credentials.token,
        )
        self.credentials.last_refresh = datetime.now(tz=UTC)
        self._persist_refreshed_chatgpt_tokens()

    def _persist_refreshed_chatgpt_tokens(self) -> None:
        if self.credentials.auth_file is None or self.credentials.auth_json is None:
            return

        auth_json = dict(self.credentials.auth_json)
        tokens = dict(auth_json.get("tokens") or {})
        tokens["access_token"] = self.credentials.token
        if self.credentials.id_token:
            tokens["id_token"] = self.credentials.id_token
        if self.credentials.refresh_token:
            tokens["refresh_token"] = self.credentials.refresh_token
        if self.credentials.account_id:
            tokens["account_id"] = self.credentials.account_id
        auth_json["tokens"] = tokens
        auth_json["last_refresh"] = _isoformat_utc(self.credentials.last_refresh or datetime.now(tz=UTC))
        self.credentials.auth_file.parent.mkdir(parents=True, exist_ok=True)
        self.credentials.auth_file.write_text(json.dumps(auth_json, indent=2))
        self.credentials.auth_json = auth_json

    def _iter_sse_events(self, response: httpx.Response) -> Iterator[SseEvent]:
        event_name: str | None = None
        data_lines: list[str] = []

        for line in response.iter_lines():
            if line == "":
                if data_lines:
                    yield self._decode_sse_event(event_name, data_lines)
                event_name = None
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line.partition(":")[2].lstrip()
                continue
            if line.startswith("data:"):
                data_lines.append(line.partition(":")[2].lstrip())
                continue

        if data_lines:
            yield self._decode_sse_event(event_name, data_lines)

    def _decode_sse_event(self, event_name: str | None, data_lines: list[str]) -> SseEvent:
        data = "\n".join(data_lines)
        json_data: dict[str, Any] | None = None
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                json_data = parsed
        except json.JSONDecodeError:
            json_data = None

        if json_data is not None:
            event_type = json_data.get("type")
            if event_type == "response.failed":
                message = "response.failed event received"
                error = json_data.get("response", {}).get("error")
                if isinstance(error, dict):
                    detail = error.get("message")
                    if isinstance(detail, str) and detail:
                        message = detail
                raise CodexResponsesError(message)
            if event_type == "response.incomplete":
                reason = (
                    json_data.get("response", {})
                    .get("incomplete_details", {})
                    .get("reason", "unknown")
                )
                raise CodexResponsesError(f"incomplete response returned, reason: {reason}")

        return SseEvent(event=event_name, data=data, json_data=json_data)

    def _raise_for_status(self, response: httpx.Response, *, prefix: str = "request failed") -> None:
        if response.is_success:
            return

        detail = response.text.strip()
        if detail:
            raise CodexResponsesError(
                f"{prefix}: {response.status_code} {response.reason_phrase}: {detail}"
            )
        raise CodexResponsesError(f"{prefix}: {response.status_code} {response.reason_phrase}")
