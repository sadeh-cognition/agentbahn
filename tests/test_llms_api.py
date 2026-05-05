from __future__ import annotations

import os
import subprocess
import sys
import types

from model_bakery import baker
from ninja.testing import TestClient

from agentbahn.api import api
from agentbahn.llms.models import decrypt_api_key
from agentbahn.llms.models import encrypt_api_key
from agentbahn.llms.models import LlmConfiguration
from agentbahn.llms.services import build_dspy_lm_from_configuration


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
        name="Groq fast",
        provider="groq",
        llm_name="llama-3.1-8b-instant",
        encrypted_api_key=encrypt_api_key("secret-key"),
    )

    response = client.get("/api/llm-config")

    assert response.status_code == 200
    assert response.json() == {
        "exists": True,
        "config": {
            "id": 1,
            "name": "Groq fast",
            "provider": "groq",
            "llm_name": "llama-3.1-8b-instant",
            "lm_backend_path": "default",
            "api_key_configured": True,
        },
    }


def test_post_llm_config_persists_encrypted_api_key(db) -> None:
    response = client.post(
        "/api/llm-config",
        json={
            "provider": "groq",
            "name": "Groq fast",
            "llm_name": "llama-3.1-8b-instant",
            "lm_backend_path": "agentbahn.llms.custom_backend",
            "api_key": "secret-key",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": 1,
        "name": "Groq fast",
        "provider": "groq",
        "llm_name": "llama-3.1-8b-instant",
        "lm_backend_path": "agentbahn.llms.custom_backend",
        "api_key_configured": True,
    }

    config = LlmConfiguration.objects.get(pk=1)
    assert config.name == "Groq fast"
    assert config.provider == "groq"
    assert config.llm_name == "llama-3.1-8b-instant"
    assert config.lm_backend_path == "agentbahn.llms.custom_backend"
    assert config.encrypted_api_key != "secret-key"
    assert config.encrypted_api_key.startswith("fernet:")
    assert decrypt_api_key(config.encrypted_api_key) == "secret-key"


def test_post_llm_config_preserves_existing_api_key_when_omitted(db) -> None:
    baker.make(
        LlmConfiguration,
        pk=1,
        name="Groq fast",
        provider="groq",
        llm_name="llama-3.1-8b-instant",
        encrypted_api_key=encrypt_api_key("secret-key"),
    )

    response = client.post(
        "/api/llm-config",
        json={
            "id": 1,
            "name": "OpenAI main",
            "provider": "openai",
            "llm_name": "gpt-5.4",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": 1,
        "name": "OpenAI main",
        "provider": "openai",
        "llm_name": "gpt-5.4",
        "lm_backend_path": "default",
        "api_key_configured": True,
    }

    config = LlmConfiguration.objects.get(pk=1)
    assert config.name == "OpenAI main"
    assert config.provider == "openai"
    assert config.llm_name == "gpt-5.4"
    assert config.lm_backend_path == "default"
    assert decrypt_api_key(config.encrypted_api_key) == "secret-key"


def test_post_llm_config_requires_api_key_for_new_configuration(db) -> None:
    response = client.post(
        "/api/llm-config",
        json={
            "provider": "groq",
            "llm_name": "llama-3.1-8b-instant",
        },
    )

    assert response.status_code == 422
    assert "API key is required" in response.json()["detail"]


def test_list_llm_configs_returns_existing_configurations(db) -> None:
    baker.make(
        LlmConfiguration,
        pk=1,
        name="Groq fast",
        provider="groq",
        llm_name="llama-3.1-8b-instant",
        encrypted_api_key=encrypt_api_key("secret-key"),
    )
    baker.make(
        LlmConfiguration,
        pk=2,
        name="OpenAI custom",
        provider="openai",
        llm_name="gpt-5.5",
        lm_backend_path="agentbahn.llms.custom_backend",
        encrypted_api_key=encrypt_api_key("other-key"),
    )

    response = client.get("/api/llm-configs")

    assert response.status_code == 200
    assert response.json() == {
        "configs": [
            {
                "id": 1,
                "name": "Groq fast",
                "provider": "groq",
                "llm_name": "llama-3.1-8b-instant",
                "lm_backend_path": "default",
                "api_key_configured": True,
            },
            {
                "id": 2,
                "name": "OpenAI custom",
                "provider": "openai",
                "llm_name": "gpt-5.5",
                "lm_backend_path": "agentbahn.llms.custom_backend",
                "api_key_configured": True,
            },
        ]
    }


def test_build_dspy_lm_from_configuration_uses_persisted_provider_model_and_key(
    db,
) -> None:
    baker.make(
        LlmConfiguration,
        pk=1,
        name="Groq fast",
        provider="groq",
        llm_name="llama-3.1-8b-instant",
        encrypted_api_key=encrypt_api_key("secret-key"),
    )

    lm = build_dspy_lm_from_configuration()

    assert lm.model == "groq/llama-3.1-8b-instant"
    assert lm.kwargs["api_key"] == "secret-key"


def test_build_dspy_lm_from_configuration_uses_custom_backend_module(db) -> None:
    module_name = "tests.dynamic_lm_backend"
    backend_module = types.ModuleType(module_name)

    class CustomLM:
        def __init__(self, model: str, api_key: str) -> None:
            self.model = model
            self.api_key = api_key

    setattr(backend_module, "LM", CustomLM)
    sys.modules[module_name] = backend_module
    baker.make(
        LlmConfiguration,
        pk=1,
        name="OpenAI custom",
        provider="openai",
        llm_name="gpt-5.5",
        lm_backend_path=module_name,
        encrypted_api_key=encrypt_api_key("secret-key"),
    )

    try:
        lm = build_dspy_lm_from_configuration()
    finally:
        del sys.modules[module_name]

    assert isinstance(lm, CustomLM)
    assert lm.model == "openai/gpt-5.5"
    assert lm.api_key == "secret-key"


def test_build_dspy_lm_from_configuration_requires_persisted_config(db) -> None:
    try:
        build_dspy_lm_from_configuration()
    except ValueError as exc:
        assert str(exc) == "No LLM configuration found."
    else:
        raise AssertionError("Expected missing LLM configuration to raise ValueError.")


def test_application_requires_llm_api_key_encryption_key() -> None:
    env = os.environ.copy()
    env["LLM_API_KEY_ENCRYPTION_KEY"] = ""

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
