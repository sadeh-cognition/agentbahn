from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentbahn.codex_openai_client import CodexCredentials
from agentbahn.codex_openai_client import CodexOpenAIClient

if TYPE_CHECKING:

    class DSPyBaseLM:
        kwargs: dict[str, Any]
        model: str

        def __init__(self, *_: Any, **__: Any) -> None: ...
else:
    try:
        from dspy import BaseLM as DSPyBaseLM
    except ModuleNotFoundError as exc:
        _DSPY_IMPORT_ERROR = exc

        class DSPyBaseLM:  # type: ignore[no-redef]
            def __init__(self, *_: Any, **__: Any) -> None:
                raise ModuleNotFoundError(
                    "CodexDSPyLM requires the 'dspy' package. Install project dependencies with uv."
                ) from _DSPY_IMPORT_ERROR


class _AttrObject(Mapping[str, Any]):
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__)

    def __len__(self) -> int:
        return len(self.__dict__)

    def model_dump(self) -> dict[str, Any]:
        return {key: _to_plain_value(value) for key, value in self.__dict__.items()}


def _to_plain_value(value: Any) -> Any:
    if isinstance(value, _AttrObject):
        return value.model_dump()
    if isinstance(value, list):
        return [_to_plain_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain_value(item) for key, item in value.items()}
    return value


def _to_attr_object(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _AttrObject(
            **{key: _to_attr_object(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_to_attr_object(item) for item in value]
    return value


def _normalize_message_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if isinstance(content, Iterable) and not isinstance(
        content, (str, bytes, bytearray)
    ):
        normalized: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, Mapping):
                raise TypeError("DSPy messages content items must be mappings")
            item_dict = dict(item)
            item_type = item_dict.get("type")
            if item_type == "text":
                normalized.append(
                    {"type": "input_text", "text": item_dict.get("text", "")}
                )
                continue
            if item_type == "image_url":
                image_url = item_dict.get("image_url")
                if isinstance(image_url, Mapping):
                    normalized.append(
                        {"type": "input_image", "image_url": image_url.get("url", "")}
                    )
                    continue
            normalized.append(item_dict)
        return normalized
    raise TypeError("DSPy messages content must be a string or a list of content items")


class CodexDSPyLM(DSPyBaseLM):
    """DSPy `BaseLM` adapter backed by `CodexOpenAIClient`."""

    def __init__(
        self,
        model: str,
        *,
        client: CodexOpenAIClient | None = None,
        credentials: CodexCredentials | None = None,
        codex_home: Path | None = None,
        enable_codex_api_key_env: bool = True,
        base_url: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = 1000,
        cache: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            model_type="responses",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            **kwargs,
        )
        self.client = client or CodexOpenAIClient(
            credentials=credentials,
            codex_home=codex_home,
            enable_codex_api_key_env=enable_codex_api_key_env,
            base_url=base_url,
        )

    @property
    def supports_function_calling(self) -> bool:
        return True

    @property
    def supports_reasoning(self) -> bool:
        return True

    @property
    def supports_response_schema(self) -> bool:
        return True

    @property
    def supported_params(self) -> set[str]:
        return {
            "frequency_penalty",
            "instructions",
            "max_output_tokens",
            "metadata",
            "parallel_tool_calls",
            "presence_penalty",
            "reasoning",
            "response_format",
            "service_tier",
            "stop",
            "temperature",
            "text",
            "tool_choice",
            "tools",
            "top_p",
            "user",
        }

    def close(self) -> None:
        self.client.close()

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        payload = self._build_payload(prompt=prompt, messages=messages, **kwargs)
        response = self.client.create_response(payload)
        return _to_attr_object(response)

    def _build_payload(
        self,
        *,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        request_kwargs = {**self.kwargs, **kwargs}
        payload: dict[str, Any] = {"model": self.model}

        resolved_messages = messages
        if resolved_messages is None:
            resolved_messages = [{"role": "user", "content": prompt or ""}]

        payload["input"] = [
            {
                "role": message.get("role", "user"),
                "content": _normalize_message_content(message.get("content", "")),
            }
            for message in resolved_messages
        ]

        max_tokens = request_kwargs.pop("max_tokens", None)
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens

        if "response_format" in request_kwargs:
            response_format = request_kwargs.pop("response_format")
            text_config = request_kwargs.pop("text", {}) or {}
            payload["text"] = {**text_config, "format": response_format}

        for key in self.supported_params:
            value = request_kwargs.pop(key, None)
            if value is not None:
                payload[key] = value

        return payload
