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
        api_key_configured=bool(config.api_key_hash),
    )


def upsert_llm_configuration(payload: LlmConfigUpsertRequest) -> LlmConfiguration:
    config, _created = LlmConfiguration.objects.update_or_create(
        pk=LLM_CONFIGURATION_PRIMARY_KEY,
        defaults={
            "provider": payload.provider,
            "llm_name": payload.llm_name,
            "api_key_hash": payload.api_key,
        },
    )
    return config
