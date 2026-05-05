from __future__ import annotations

import importlib
from typing import cast

import dspy

from agentbahn.llms.models import decrypt_api_key
from agentbahn.llms.models import LlmConfiguration
from agentbahn.llms.schemas import LlmConfigListResponse
from agentbahn.llms.schemas import LlmConfigResponse
from agentbahn.llms.schemas import LlmConfigUpsertRequest


def get_llm_configuration() -> LlmConfiguration | None:
    return LlmConfiguration.objects.order_by("id").first()


def list_llm_configurations() -> LlmConfigListResponse:
    return LlmConfigListResponse(
        configs=[
            serialize_llm_configuration(config)
            for config in LlmConfiguration.objects.order_by("id")
        ]
    )


def build_dspy_lm_from_configuration(
    config: LlmConfiguration | None = None,
) -> dspy.BaseLM:
    runtime_config = config or get_llm_configuration()
    if runtime_config is None:
        raise ValueError("No LLM configuration found.")
    api_key = decrypt_api_key(runtime_config.encrypted_api_key)
    if not api_key:
        raise ValueError("LLM API key is not configured.")
    lm_class = dspy.LM
    if runtime_config.lm_backend_path != "default":
        lm_backend_module = importlib.import_module(runtime_config.lm_backend_path)
        lm_class = cast(type[dspy.BaseLM], getattr(lm_backend_module, "LM"))
    return lm_class(
        model=f"{runtime_config.provider}/{runtime_config.llm_name}",
        api_key=api_key,
    )


def serialize_llm_configuration(config: LlmConfiguration) -> LlmConfigResponse:
    return LlmConfigResponse(
        id=config.id,
        name=config.name,
        provider=config.provider,
        llm_name=config.llm_name,
        lm_backend_path=config.lm_backend_path,
        api_key_configured=bool(config.encrypted_api_key),
    )


def upsert_llm_configuration(payload: LlmConfigUpsertRequest) -> LlmConfiguration:
    config = (
        LlmConfiguration.objects.filter(pk=payload.id).first()
        if payload.id is not None
        else None
    )
    if config is None:
        if payload.api_key is None:
            raise ValueError("API key is required when creating LLM configuration.")
        config = LlmConfiguration()

    config.name = payload.name or f"{payload.provider}/{payload.llm_name}"
    config.provider = payload.provider
    config.llm_name = payload.llm_name
    config.lm_backend_path = payload.lm_backend_path
    if payload.api_key is not None:
        config.encrypted_api_key = payload.api_key
    config.save()
    return config
