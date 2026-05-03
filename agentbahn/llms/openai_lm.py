from __future__ import annotations

from typing import Any

import dspy
from openai import AsyncOpenAI
from openai import OpenAI


DEFAULT_FLEX_TIMEOUT_SECONDS = 900.0


class OpenAIFlexLM(dspy.BaseLM):
    """DSPy LM backed by the official OpenAI SDK with Flex processing enabled."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        timeout: float = DEFAULT_FLEX_TIMEOUT_SECONDS,
        client: Any | None = None,
        async_client: Any | None = None,
        service_tier: str = "flex",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, model_type="chat", **kwargs)
        self.api_key = api_key
        self.timeout = timeout
        self.service_tier = service_tier
        self.client = client or OpenAI(api_key=api_key, timeout=timeout)
        self.async_client = async_client or AsyncOpenAI(
            api_key=api_key, timeout=timeout
        )

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        request_kwargs = self._build_request_kwargs(prompt, messages, kwargs)
        return self.client.with_options(timeout=self.timeout).chat.completions.create(
            **request_kwargs
        )

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        request_kwargs = self._build_request_kwargs(prompt, messages, kwargs)
        return await self.async_client.with_options(
            timeout=self.timeout
        ).chat.completions.create(**request_kwargs)

    def _build_request_kwargs(
        self,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        request_messages = messages or [{"role": "user", "content": prompt}]
        request_kwargs = {
            **self.kwargs,
            **kwargs,
            "model": self.model,
            "messages": request_messages,
            "service_tier": self.service_tier,
        }
        if request_kwargs.get("rollout_id") is None:
            request_kwargs.pop("rollout_id", None)
        return request_kwargs
