from __future__ import annotations

from typing import Literal

from ninja import Schema
from pydantic import field_validator


class CodebaseAgentRequest(Schema):
    query: str
    llm_config_id: int | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("Query cannot be blank.")
        return normalized_value


class CodebaseAgentResponse(Schema):
    result: str


class CodebaseAgentStreamEvent(Schema):
    type: Literal["token", "result", "error"]
    content: str | None = None
    detail: str | None = None
