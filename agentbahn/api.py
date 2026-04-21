from ninja import NinjaAPI

from agentbahn.llms.api import router as llm_router

api = NinjaAPI()
api.add_router("/api/", llm_router)
