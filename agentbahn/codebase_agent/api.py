from __future__ import annotations

from ninja import Router
from ninja.errors import HttpError

from agentbahn.codebase_agent.schemas import CodebaseAgentRequest
from agentbahn.codebase_agent.schemas import CodebaseAgentResponse
from agentbahn.codebase_agent.services import run_codebase_agent

router = Router(tags=["codebase-agent"])


@router.post("/codebase-agent", response=CodebaseAgentResponse)
def codebase_agent(
    request,
    payload: CodebaseAgentRequest,
) -> CodebaseAgentResponse:
    del request
    try:
        result = run_codebase_agent(payload.query)
    except ValueError as exc:
        raise HttpError(422, str(exc)) from exc
    return CodebaseAgentResponse(result=result)
