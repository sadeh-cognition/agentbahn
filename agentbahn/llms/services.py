from __future__ import annotations

from agentbahn.llms.models import LlmConfiguration
from agentbahn.llms.schemas import LlmConfigResponse
from agentbahn.llms.schemas import LlmConfigUpsertRequest

LLM_CONFIGURATION_PRIMARY_KEY = 1


def get_llm_configuration() -> LlmConfiguration | None:
    return LlmConfiguration.objects.filter(pk=LLM_CONFIGURATION_PRIMARY_KEY).first()


def serialize_llm_configuration(config: LlmConfiguration) -> LlmConfigResponse:
    return LlmConfigResponse(
        provider=config.provider,
        llm_name=config.llm_name,
        api_key_configured=bool(config.encrypted_api_key),
    )


def upsert_llm_configuration(payload: LlmConfigUpsertRequest) -> LlmConfiguration:
    config = get_llm_configuration()
    if config is None:
        if payload.api_key is None:
            raise ValueError("API key is required when creating LLM configuration.")
        config = LlmConfiguration(pk=LLM_CONFIGURATION_PRIMARY_KEY)

    config.provider = payload.provider
    config.llm_name = payload.llm_name
    if payload.api_key is not None:
        config.encrypted_api_key = payload.api_key
    config.save()
    return config
