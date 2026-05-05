from __future__ import annotations

import asyncio

import httpx

from agentbahn.codebase_agent.schemas import CodebaseAgentStreamEvent
from agentbahn.llms.schemas import LlmConfigListResponse
from agentbahn.llms.schemas import LlmConfigLookupResponse
from agentbahn.llms.schemas import LlmConfigResponse
from agentbahn.llms.schemas import LlmConfigUpsertRequest
from agentbahn_tui.backend import BackendUnavailableError
from agentbahn_tui.backend import check_backend_server_running
from agentbahn_tui.llm_commands import continue_llm_configuration
from agentbahn_tui.llm_commands import start_llm_command
from agentbahn_tui.llms import fetch_llm_config
from agentbahn_tui.llms import fetch_llm_configs
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
                    "name": "Groq fast",
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
            name="Groq fast",
            provider="groq",
            llm_name="llama-3.1-8b-instant",
            api_key_configured=True,
        ),
    )


def test_fetch_llm_configs_uses_agentbahn_api_base_url(settings) -> None:
    settings.API_BASE_URL = "http://testserver"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/llm-configs"
        return httpx.Response(
            200,
            json={
                "configs": [
                    {
                        "id": 1,
                        "name": "Groq fast",
                        "provider": "groq",
                        "llm_name": "llama-3.1-8b-instant",
                        "api_key_configured": True,
                    }
                ]
            },
        )

    response = fetch_llm_configs(transport=httpx.MockTransport(handler))

    assert response == LlmConfigListResponse(
        configs=[
            LlmConfigResponse(
                id=1,
                name="Groq fast",
                provider="groq",
                llm_name="llama-3.1-8b-instant",
                api_key_configured=True,
            )
        ]
    )


def test_save_llm_config_posts_schema_payload(settings) -> None:
    settings.API_BASE_URL = "http://testserver"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/llm-config"
        assert request.method == "POST"
        assert request.content == (
            b'{"name":"Groq fast","provider":"groq","llm_name":"llama-3.1-8b-instant","api_key":"secret-key"}'
        )
        return httpx.Response(
            200,
            json={
                "name": "Groq fast",
                "provider": "groq",
                "llm_name": "llama-3.1-8b-instant",
                "api_key_configured": True,
            },
        )

    response = save_llm_config(
        LlmConfigUpsertRequest(
            name="Groq fast",
            provider="groq",
            llm_name="llama-3.1-8b-instant",
            api_key="secret-key",
        ),
        transport=httpx.MockTransport(handler),
    )

    assert response == LlmConfigResponse(
        name="Groq fast",
        provider="groq",
        llm_name="llama-3.1-8b-instant",
        api_key_configured=True,
    )


def test_save_llm_config_omits_absent_api_key(settings) -> None:
    settings.API_BASE_URL = "http://testserver"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/llm-config"
        assert request.method == "POST"
        assert (
            request.content
            == b'{"name":"OpenAI main","provider":"openai","llm_name":"gpt-5.4"}'
        )
        return httpx.Response(
            200,
            json={
                "name": "OpenAI main",
                "provider": "openai",
                "llm_name": "gpt-5.4",
                "api_key_configured": True,
            },
        )

    response = save_llm_config(
        LlmConfigUpsertRequest(
            name="OpenAI main",
            provider="openai",
            llm_name="gpt-5.4",
        ),
        transport=httpx.MockTransport(handler),
    )

    assert response == LlmConfigResponse(
        name="OpenAI main",
        provider="openai",
        llm_name="gpt-5.4",
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
                name="Groq fast",
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
        "ID: 0\n"
        "Name: Groq fast\n"
        "Provider: groq\n"
        "LLM name: llama-3.1-8b-instant\n"
        "LM backend path: default\n"
        "API key: configured"
    )


def test_continue_llm_configuration_collects_and_saves_values() -> None:
    name_step = continue_llm_configuration(
        state=start_llm_command(
            lambda: LlmConfigLookupResponse(exists=False)
        ).next_state,
        value="Groq fast",
        save_llm_config_command=lambda payload: LlmConfigResponse(
            name=payload.name or "",
            provider=payload.provider,
            llm_name=payload.llm_name,
            api_key_configured=bool(payload.api_key),
        ),
    )

    assert name_step.next_state is not None
    assert name_step.next_state.name == "Groq fast"
    assert name_step.next_state.step == "provider"

    provider_step = continue_llm_configuration(
        state=name_step.next_state,
        value="groq",
        save_llm_config_command=lambda payload: LlmConfigResponse(
            name=payload.name or "",
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
            name=payload.name or "",
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
                name=payload.name or "",
                provider=payload.provider,
                llm_name=payload.llm_name,
                api_key_configured=True,
            )
        ),
    )

    assert captured_payloads == [
        LlmConfigUpsertRequest(
            name="Groq fast",
            provider="groq",
            llm_name="llama-3.1-8b-instant",
            api_key="secret-key",
        )
    ]
    assert saved_step.next_state is None
    assert saved_step.secret_input is False


def test_tui_llm_command_shows_editable_form_and_avoids_storing_api_key_in_history(
    tmp_path,
) -> None:
    async def run_test() -> None:
        history_file = tmp_path / "command_history"
        captured_payloads: list[LlmConfigUpsertRequest] = []
        app = AgentbahnTui(
            fetch_projects_command=lambda: [],
            fetch_tasks_command=lambda _project_id: [],
            fetch_project_events_command=lambda _project_id: [],
            fetch_llm_configs_command=lambda: LlmConfigListResponse(configs=[]),
            save_llm_config_command=lambda payload: (
                captured_payloads.append(payload)
                or LlmConfigResponse(
                    id=1,
                    name=payload.name or "",
                    provider=payload.provider,
                    llm_name=payload.llm_name,
                    api_key_configured=bool(payload.api_key),
                )
            ),
            history_file=history_file,
        )

        async with app.run_test() as pilot:
            await pilot.press("/", "l", "l", "m")
            await pilot.press("enter")
            assert app.query_one("#llm-config-form").display is True

            await pilot.press("G", "r", "o", "q", " ", "f", "a", "s", "t")
            await pilot.press("tab")

            await pilot.press("g", "r", "o", "q")
            await pilot.press("tab")

            await pilot.press("l", "l", "a", "m", "a")
            await pilot.press("tab")
            await pilot.press("tab")

            await pilot.press("s", "e", "c", "r", "e", "t")
            await pilot.press("enter")
            assert captured_payloads == [
                LlmConfigUpsertRequest(
                    name="Groq fast",
                    provider="groq",
                    llm_name="llama",
                    api_key="secret",
                )
            ]
            assert "Created LLM configuration." in str(
                app.query_one("#llm-form-status").content
            )

        assert history_file.read_text(encoding="utf-8") == "/llm\n"

    asyncio.run(run_test())


def test_tui_llm_form_lists_existing_configs_and_creates_new_config(
    tmp_path,
) -> None:
    async def run_test() -> None:
        captured_payloads: list[LlmConfigUpsertRequest] = []
        app = AgentbahnTui(
            fetch_projects_command=lambda: [],
            fetch_tasks_command=lambda _project_id: [],
            fetch_project_events_command=lambda _project_id: [],
            fetch_llm_configs_command=lambda: LlmConfigListResponse(
                configs=[
                    LlmConfigResponse(
                        id=1,
                        name="Groq fast",
                        provider="groq",
                        llm_name="llama-3.1-8b-instant",
                        api_key_configured=True,
                    )
                ],
            ),
            save_llm_config_command=lambda payload: (
                captured_payloads.append(payload)
                or LlmConfigResponse(
                    id=2,
                    name=payload.name or "",
                    provider=payload.provider,
                    llm_name=payload.llm_name,
                    api_key_configured=True,
                )
            ),
            history_file=tmp_path / "command_history",
        )

        async with app.run_test() as pilot:
            await pilot.press("/", "l", "l", "m")
            await pilot.press("enter")
            friendly_name_input = app.query_one("#llm-friendly-name-input")
            provider_input = app.query_one("#llm-provider-input")
            llm_name_input = app.query_one("#llm-name-input")
            lm_backend_path_input = app.query_one("#llm-backend-path-input")
            lm_backend_path_tip = app.query_one("#llm-backend-path-tip")
            api_key_input = app.query_one("#llm-api-key-input")
            llm_config_list = app.query_one("#llm-config-list")

            assert friendly_name_input.value == ""
            assert provider_input.value == ""
            assert llm_name_input.value == ""
            assert lm_backend_path_input.value == "default"
            assert "1. Groq fast - groq/llama-3.1-8b-instant" in str(
                llm_config_list.content
            )
            assert "custom DSPy LM backend module" in str(lm_backend_path_tip.content)
            assert "'default' uses DSPy's standard LM backend" in str(
                lm_backend_path_tip.content
            )
            assert api_key_input.value == ""

            friendly_name_input.value = "OpenAI main"
            provider_input.value = "openai"
            llm_name_input.value = "gpt-5.4"
            lm_backend_path_input.value = "agentbahn.llms.custom_backend"
            api_key_input.value = "new-secret"
            await pilot.press("enter")

        assert captured_payloads == [
            LlmConfigUpsertRequest(
                name="OpenAI main",
                provider="openai",
                llm_name="gpt-5.4",
                lm_backend_path="agentbahn.llms.custom_backend",
                api_key="new-secret",
            )
        ]

    asyncio.run(run_test())


def test_tui_llm_form_selects_existing_config_for_update(tmp_path) -> None:
    async def run_test() -> None:
        captured_payloads: list[LlmConfigUpsertRequest] = []
        saved_response = LlmConfigResponse(
            id=1,
            name="OpenAI main",
            provider="openai",
            llm_name="gpt-5.4",
            lm_backend_path="agentbahn.llms.custom_backend",
            api_key_configured=True,
        )
        configs = [
            LlmConfigResponse(
                id=1,
                name="Groq fast",
                provider="groq",
                llm_name="llama-3.1-8b-instant",
                api_key_configured=True,
            )
        ]
        app = AgentbahnTui(
            fetch_projects_command=lambda: [],
            fetch_tasks_command=lambda _project_id: [],
            fetch_project_events_command=lambda _project_id: [],
            fetch_llm_configs_command=lambda: LlmConfigListResponse(configs=configs),
            save_llm_config_command=lambda payload: (
                captured_payloads.append(payload) or saved_response
            ),
            history_file=tmp_path / "command_history",
        )

        async with app.run_test() as pilot:
            await pilot.press("/", "l", "l", "m")
            await pilot.press("enter")
            llm_config_select = app.query_one("#llm-config-select")
            friendly_name_input = app.query_one("#llm-friendly-name-input")
            provider_input = app.query_one("#llm-provider-input")
            llm_name_input = app.query_one("#llm-name-input")
            lm_backend_path_input = app.query_one("#llm-backend-path-input")
            api_key_input = app.query_one("#llm-api-key-input")
            save_button = app.query_one("#llm-save-button")

            llm_config_select.value = "1"
            await pilot.pause()

            assert friendly_name_input.value == "Groq fast"
            assert provider_input.value == "groq"
            assert llm_name_input.value == "llama-3.1-8b-instant"
            assert lm_backend_path_input.value == "default"
            assert api_key_input.value == ""
            assert api_key_input.placeholder == "Leave blank to keep configured key"
            assert str(save_button.label) == "Update LLM configuration"

            friendly_name_input.value = "OpenAI main"
            provider_input.value = "openai"
            llm_name_input.value = "gpt-5.4"
            lm_backend_path_input.value = "agentbahn.llms.custom_backend"
            api_key_input.value = ""
            await pilot.press("enter")

            assert captured_payloads == [
                LlmConfigUpsertRequest(
                    id=1,
                    name="OpenAI main",
                    provider="openai",
                    llm_name="gpt-5.4",
                    lm_backend_path="agentbahn.llms.custom_backend",
                )
            ]
            assert "Updated LLM configuration." in str(
                app.query_one("#llm-form-status").content
            )

    asyncio.run(run_test())


def test_tui_model_command_selects_config_for_agent_chat(tmp_path) -> None:
    async def run_test() -> None:
        captured_requests: list[tuple[str, int | None]] = []
        configs = [
            LlmConfigResponse(
                id=1,
                name="Groq fast",
                provider="groq",
                llm_name="llama-3.1-8b-instant",
                api_key_configured=True,
            ),
            LlmConfigResponse(
                id=2,
                name="OpenAI main",
                provider="openai",
                llm_name="gpt-5.5",
                api_key_configured=True,
            ),
        ]

        def stream_agent_command(query: str, llm_config_id: int | None):
            captured_requests.append((query, llm_config_id))
            yield CodebaseAgentStreamEvent(type="result", content="Done")

        app = AgentbahnTui(
            fetch_projects_command=lambda: [],
            fetch_tasks_command=lambda _project_id: [],
            fetch_project_events_command=lambda _project_id: [],
            fetch_llm_configs_command=lambda: LlmConfigListResponse(configs=configs),
            stream_agent_command=stream_agent_command,
            history_file=tmp_path / "command_history",
        )

        async with app.run_test() as pilot:
            await pilot.press("/", "m", "o", "d", "e", "l")
            await pilot.press("enter")
            model_select = app.query_one("#model-config-select")
            model_table = app.query_one("#model-config-table")
            assert model_table.row_count == 2
            assert model_table.get_row_at(0)[0] == ""
            assert model_table.get_row_at(1)[0] == ""
            model_select.value = "2"
            await pilot.pause()
            assert model_table.get_row_at(1)[0] == ""
            await pilot.click("#model-use-button")
            assert "Selected model for codebase-agent chat" in str(
                app.query_one("#model-status").content
            )
            assert model_table.get_row_at(0)[0] == ""
            assert model_table.get_row_at(1)[0] == "✓"

            command_input = app.query_one("#command-input")
            command_input.focus()
            await pilot.press("B", "u", "i", "l", "d")
            await pilot.press("enter")
            await pilot.pause()

        assert captured_requests == [("Build", 2)]

    asyncio.run(run_test())
