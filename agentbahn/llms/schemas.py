from __future__ import annotations

from ninja import Schema
from pydantic import field_validator


class LlmConfigResponse(Schema):
    provider: str
    llm_name: str
    api_key_configured: bool


class LlmConfigLookupResponse(Schema):
    exists: bool
    config: LlmConfigResponse | None = None


class LlmConfigUpsertRequest(Schema):
    provider: str
    llm_name: str
    api_key: str

    @field_validator("provider", "llm_name", "api_key")
    @classmethod
    def validate_not_blank(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("This field cannot be blank.")
        return normalized_value
