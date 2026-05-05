from __future__ import annotations

from django.http import StreamingHttpResponse
from ninja import Router
from ninja.errors import HttpError

from agentbahn.codebase_agent.schemas import CodebaseAgentRequest
from agentbahn.codebase_agent.services import async_stream_codebase_agent
from agentbahn.codebase_agent.services import validate_llm_configuration_id_async

router = Router(tags=["codebase-agent"])


@router.post("/codebase-agent")
async def codebase_agent(
    request,
    payload: CodebaseAgentRequest,
) -> StreamingHttpResponse:
    del request
    try:
        await validate_llm_configuration_id_async(payload.llm_config_id)
        stream = async_stream_codebase_agent(
            payload.query,
            llm_config_id=payload.llm_config_id,
        )
    except ValueError as exc:
        raise HttpError(422, str(exc)) from exc
    return StreamingHttpResponse(
        stream,
        content_type="application/x-ndjson; charset=utf-8",
    )
