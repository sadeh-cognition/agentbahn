from __future__ import annotations

from django.contrib.auth.hashers import check_password
from model_bakery import baker
from ninja.testing import TestClient

from agentbahn.api import api
from agentbahn.llms.models import LlmConfiguration


client = TestClient(api)


def test_get_llm_config_returns_absent_state(db) -> None:
    response = client.get("/api/llm-config")

    assert response.status_code == 200
    assert response.json() == {"exists": False, "config": None}


def test_get_llm_config_returns_existing_configuration(db) -> None:
    baker.make(
        LlmConfiguration,
        pk=1,
        provider="groq",
        llm_name="llama-3.1-8b-instant",
        api_key_hash="pbkdf2_sha256$1000000$salt$hash",
    )

    response = client.get("/api/llm-config")

    assert response.status_code == 200
    assert response.json() == {
        "exists": True,
        "config": {
            "provider": "groq",
            "llm_name": "llama-3.1-8b-instant",
            "api_key_configured": True,
        },
    }


def test_post_llm_config_persists_hashed_api_key(db) -> None:
    response = client.post(
        "/api/llm-config",
        json={
            "provider": "groq",
            "llm_name": "llama-3.1-8b-instant",
            "api_key": "secret-key",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "provider": "groq",
        "llm_name": "llama-3.1-8b-instant",
        "api_key_configured": True,
    }

    config = LlmConfiguration.objects.get(pk=1)
    assert config.provider == "groq"
    assert config.llm_name == "llama-3.1-8b-instant"
    assert config.api_key_hash != "secret-key"
    assert check_password("secret-key", config.api_key_hash)
