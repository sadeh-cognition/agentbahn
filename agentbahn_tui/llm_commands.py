from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from agentbahn.llms.schemas import LlmConfigLookupResponse
from agentbahn.llms.schemas import LlmConfigResponse
from agentbahn.llms.schemas import LlmConfigUpsertRequest
from agentbahn_tui.command_results import CommandResult
from agentbahn_tui.command_results import message_result


PromptStep = Literal["provider", "llm_name", "api_key"]


@dataclass(frozen=True)
class LlmConfigurationPromptState:
    provider: str | None = None
    llm_name: str | None = None
    step: PromptStep = "provider"


@dataclass(frozen=True)
class LlmCommandTransition:
    result: CommandResult
    next_state: LlmConfigurationPromptState | None = None
    secret_input: bool = False


def format_llm_configuration(config: LlmConfigResponse) -> str:
    api_key_status = "configured" if config.api_key_configured else "missing"
    return (
        "Current LLM configuration:\n"
        f"ID: {config.id}\n"
        f"Provider: {config.provider}\n"
        f"LLM name: {config.llm_name}\n"
        f"LM backend path: {config.lm_backend_path}\n"
        f"API key: {api_key_status}"
    )


def start_llm_command(
    fetch_llm_config_command: Callable[[], LlmConfigLookupResponse],
) -> LlmCommandTransition:
    lookup_response = fetch_llm_config_command()
    if lookup_response.exists and lookup_response.config is not None:
        return LlmCommandTransition(
            result=message_result(format_llm_configuration(lookup_response.config))
        )
    return LlmCommandTransition(
        result=message_result("No LLM configuration found.\nEnter provider:"),
        next_state=LlmConfigurationPromptState(step="provider"),
    )


def continue_llm_configuration(
    state: LlmConfigurationPromptState,
    value: str,
    save_llm_config_command: Callable[[LlmConfigUpsertRequest], LlmConfigResponse],
) -> LlmCommandTransition:
    normalized_value = value.strip()
    if not normalized_value:
        return LlmCommandTransition(
            result=message_result(_build_blank_value_message(state.step)),
            next_state=state,
            secret_input=state.step == "api_key",
        )

    if state.step == "provider":
        return LlmCommandTransition(
            result=message_result("Enter LLM name:"),
            next_state=LlmConfigurationPromptState(
                provider=normalized_value,
                step="llm_name",
            ),
        )

    if state.step == "llm_name":
        return LlmCommandTransition(
            result=message_result("Enter API key:"),
            next_state=LlmConfigurationPromptState(
                provider=state.provider,
                llm_name=normalized_value,
                step="api_key",
            ),
            secret_input=True,
        )

    saved_config = save_llm_config_command(
        LlmConfigUpsertRequest(
            provider=state.provider or "",
            llm_name=state.llm_name or "",
            api_key=normalized_value,
        )
    )
    return LlmCommandTransition(
        result=message_result(
            "Saved LLM configuration.\n" + format_llm_configuration(saved_config)
        )
    )


def _build_blank_value_message(step: PromptStep) -> str:
    if step == "provider":
        return "Provider cannot be empty.\nEnter provider:"
    if step == "llm_name":
        return "LLM name cannot be empty.\nEnter LLM name:"
    return "API key cannot be empty.\nEnter API key:"
