from ninja import NinjaAPI
from pydantic import BaseModel

from agentbahn.codebase_agent.api import router as codebase_agent_router
from agentbahn.llms.api import router as llm_router


class HealthResponse(BaseModel):
    status: str


api = NinjaAPI()
api.add_router("/api/", codebase_agent_router)
api.add_router("/api/", llm_router)


@api.get("/api/health", response=HealthResponse)
def health(request) -> HealthResponse:
    return HealthResponse(status="ok")
