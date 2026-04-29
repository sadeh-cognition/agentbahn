from __future__ import annotations

from ninja import Router
from ninja.errors import HttpError

from agentbahn.llms.schemas import LlmConfigLookupResponse
from agentbahn.llms.schemas import LlmConfigResponse
from agentbahn.llms.schemas import LlmConfigUpsertRequest
from agentbahn.llms.services import get_llm_configuration
from agentbahn.llms.services import serialize_llm_configuration
from agentbahn.llms.services import upsert_llm_configuration

router = Router(tags=["llm"])


@router.get("/llm-config", response=LlmConfigLookupResponse)
def get_llm_config(request) -> LlmConfigLookupResponse:
    del request
    config = get_llm_configuration()
    if config is None:
        return LlmConfigLookupResponse(exists=False)
    return LlmConfigLookupResponse(
        exists=True,
        config=serialize_llm_configuration(config),
    )


@router.post("/llm-config", response=LlmConfigResponse)
def save_llm_config(request, payload: LlmConfigUpsertRequest) -> LlmConfigResponse:
    del request
    try:
        config = upsert_llm_configuration(payload)
    except ValueError as exc:
        raise HttpError(422, str(exc)) from exc
    return serialize_llm_configuration(config)
