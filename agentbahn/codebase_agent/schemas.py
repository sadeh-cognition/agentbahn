from __future__ import annotations

from ninja import Schema
from pydantic import field_validator


class CodebaseAgentRequest(Schema):
    query: str

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("Query cannot be blank.")
        return normalized_value


class CodebaseAgentResponse(Schema):
    result: str
