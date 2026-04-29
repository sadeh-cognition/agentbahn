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
    api_key: str | None = None

    @field_validator("provider", "llm_name")
    @classmethod
    def validate_not_blank(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("This field cannot be blank.")
        return normalized_value

    @field_validator("api_key")
    @classmethod
    def normalize_api_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized_value = value.strip()
        return normalized_value or None
