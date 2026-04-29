from __future__ import annotations

import os
import subprocess
import sys

from model_bakery import baker
from ninja.testing import TestClient

from agentbahn.api import api
from agentbahn.llms.models import decrypt_api_key
from agentbahn.llms.models import encrypt_api_key
from agentbahn.llms.models import LlmConfiguration


client = TestClient(api)


def test_health_endpoint_reports_ok() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
        encrypted_api_key=encrypt_api_key("secret-key"),
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


def test_post_llm_config_persists_encrypted_api_key(db) -> None:
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
    assert config.encrypted_api_key != "secret-key"
    assert config.encrypted_api_key.startswith("fernet:")
    assert decrypt_api_key(config.encrypted_api_key) == "secret-key"


def test_application_requires_llm_api_key_encryption_key() -> None:
    env = os.environ.copy()
    env.pop("LLM_API_KEY_ENCRYPTION_KEY", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import agentbahn.settings",
        ],
        cwd=os.getcwd(),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "LLM_API_KEY_ENCRYPTION_KEY environment variable must be set." in (
        result.stderr + result.stdout
    )
