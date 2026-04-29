from __future__ import annotations

import asyncio

import httpx

from agentbahn.llms.schemas import LlmConfigLookupResponse
from agentbahn.llms.schemas import LlmConfigResponse
from agentbahn.llms.schemas import LlmConfigUpsertRequest
from agentbahn_tui.backend import BackendUnavailableError
from agentbahn_tui.backend import check_backend_server_running
from agentbahn_tui.llm_commands import continue_llm_configuration
from agentbahn_tui.llm_commands import start_llm_command
from agentbahn_tui.llms import fetch_llm_config
from agentbahn_tui.llms import save_llm_config
from agentbahn_tui.tui import AgentbahnTui


def test_fetch_llm_config_uses_agentbahn_api_base_url(settings) -> None:
    settings.API_BASE_URL = "http://testserver"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/llm-config"
        return httpx.Response(
            200,
            json={
                "exists": True,
                "config": {
                    "provider": "groq",
                    "llm_name": "llama-3.1-8b-instant",
                    "api_key_configured": True,
                },
            },
        )

    response = fetch_llm_config(transport=httpx.MockTransport(handler))

    assert response == LlmConfigLookupResponse(
        exists=True,
        config=LlmConfigResponse(
            provider="groq",
            llm_name="llama-3.1-8b-instant",
            api_key_configured=True,
        ),
    )


def test_save_llm_config_posts_schema_payload(settings) -> None:
    settings.API_BASE_URL = "http://testserver"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/llm-config"
        assert request.method == "POST"
        assert request.content == (
            b'{"provider":"groq","llm_name":"llama-3.1-8b-instant","api_key":"secret-key"}'
        )
        return httpx.Response(
            200,
            json={
                "provider": "groq",
                "llm_name": "llama-3.1-8b-instant",
                "api_key_configured": True,
            },
        )

    response = save_llm_config(
        LlmConfigUpsertRequest(
            provider="groq",
            llm_name="llama-3.1-8b-instant",
            api_key="secret-key",
        ),
        transport=httpx.MockTransport(handler),
    )

    assert response == LlmConfigResponse(
        provider="groq",
        llm_name="llama-3.1-8b-instant",
        api_key_configured=True,
    )


def test_check_backend_server_running_uses_health_endpoint(settings) -> None:
    settings.API_BASE_URL = "http://testserver"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/health"
        return httpx.Response(200, json={"status": "ok"})

    check_backend_server_running(transport=httpx.MockTransport(handler))


def test_check_backend_server_running_raises_when_unavailable(settings) -> None:
    settings.API_BASE_URL = "http://testserver"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    try:
        check_backend_server_running(transport=httpx.MockTransport(handler))
    except BackendUnavailableError as exc:
        assert "http://testserver" in str(exc)
    else:
        raise AssertionError("Expected BackendUnavailableError")


def test_start_llm_command_shows_existing_configuration() -> None:
    transition = start_llm_command(
        lambda: LlmConfigLookupResponse(
            exists=True,
            config=LlmConfigResponse(
                provider="groq",
                llm_name="llama-3.1-8b-instant",
                api_key_configured=True,
            ),
        )
    )

    assert transition.next_state is None
    assert transition.secret_input is False
    assert transition.result.message == (
        "Current LLM configuration:\n"
        "Provider: groq\n"
        "LLM name: llama-3.1-8b-instant\n"
        "API key: configured"
    )


def test_continue_llm_configuration_collects_and_saves_values() -> None:
    provider_step = continue_llm_configuration(
        state=start_llm_command(
            lambda: LlmConfigLookupResponse(exists=False)
        ).next_state,
        value="groq",
        save_llm_config_command=lambda payload: LlmConfigResponse(
            provider=payload.provider,
            llm_name=payload.llm_name,
            api_key_configured=bool(payload.api_key),
        ),
    )

    assert provider_step.next_state is not None
    assert provider_step.next_state.provider == "groq"
    assert provider_step.next_state.step == "llm_name"

    llm_name_step = continue_llm_configuration(
        state=provider_step.next_state,
        value="llama-3.1-8b-instant",
        save_llm_config_command=lambda payload: LlmConfigResponse(
            provider=payload.provider,
            llm_name=payload.llm_name,
            api_key_configured=bool(payload.api_key),
        ),
    )

    assert llm_name_step.next_state is not None
    assert llm_name_step.next_state.llm_name == "llama-3.1-8b-instant"
    assert llm_name_step.secret_input is True

    captured_payloads: list[LlmConfigUpsertRequest] = []
    saved_step = continue_llm_configuration(
        state=llm_name_step.next_state,
        value="secret-key",
        save_llm_config_command=lambda payload: (
            captured_payloads.append(payload)
            or LlmConfigResponse(
                provider=payload.provider,
                llm_name=payload.llm_name,
                api_key_configured=True,
            )
        ),
    )

    assert captured_payloads == [
        LlmConfigUpsertRequest(
            provider="groq",
            llm_name="llama-3.1-8b-instant",
            api_key="secret-key",
        )
    ]
    assert saved_step.next_state is None
    assert saved_step.secret_input is False


def test_tui_llm_command_prompts_and_avoids_storing_api_key_in_history(
    tmp_path,
) -> None:
    async def run_test() -> None:
        history_file = tmp_path / "command_history"
        app = AgentbahnTui(
            fetch_projects_command=lambda: [],
            fetch_tasks_command=lambda _project_id: [],
            fetch_project_events_command=lambda _project_id: [],
            fetch_llm_config_command=lambda: LlmConfigLookupResponse(exists=False),
            save_llm_config_command=lambda payload: LlmConfigResponse(
                provider=payload.provider,
                llm_name=payload.llm_name,
                api_key_configured=bool(payload.api_key),
            ),
            history_file=history_file,
        )

        async with app.run_test() as pilot:
            await pilot.press("/", "l", "l", "m")
            await pilot.press("enter")
            assert "Enter provider:" in str(app.query_one("#message-output").content)

            await pilot.press("g", "r", "o", "q")
            await pilot.press("enter")
            assert "Enter LLM name:" in str(app.query_one("#message-output").content)

            await pilot.press("l", "l", "a", "m", "a")
            await pilot.press("enter")
            command_input = app.query_one("#command-input")
            assert command_input.password is True

            await pilot.press("s", "e", "c", "r", "e", "t")
            await pilot.press("enter")
            assert command_input.password is False
            assert "Saved LLM configuration." in str(
                app.query_one("#message-output").content
            )

        assert history_file.read_text(encoding="utf-8") == "/llm\n"

    asyncio.run(run_test())
