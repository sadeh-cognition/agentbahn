from __future__ import annotations

import json
import queue
from collections.abc import AsyncIterator
from typing import Protocol

from asgiref.sync import sync_to_async
from django.conf import settings
import dspy
from dspy.streaming import StreamResponse

from agentbahn.codebase_agent.agent import AgentConfig
from agentbahn.codebase_agent.agent import DefaultAgent
from agentbahn.codebase_agent.environment import LocalEnvironment
from agentbahn.codebase_agent.environment import LocalEnvironmentConfig
from agentbahn.llms.models import LlmConfiguration
from agentbahn.llms.services import build_dspy_lm_from_configuration


class CodebaseAgent(Protocol):
    def run(self, task: str) -> dspy.Prediction: ...

    def stream(
        self,
        task: str,
    ) -> AsyncIterator[StreamResponse | dspy.Prediction]: ...

    def add_messages(self, *messages: dict) -> list[dict]: ...

    def _prediction_to_message(self, prediction: dspy.Prediction) -> dict: ...


def async_stream_codebase_agent(
    query: str,
    *,
    llm_config_id: int | None = None,
) -> AsyncIterator[str]:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Query cannot be blank.")

    return _async_stream_codebase_agent(
        normalized_query,
        llm_config_id=llm_config_id,
    )


def _validate_llm_configuration_id(llm_config_id: int) -> None:
    if not LlmConfiguration.objects.filter(pk=llm_config_id).exists():
        raise ValueError(f"LLM configuration {llm_config_id} was not found.")


async def validate_llm_configuration_id_async(
    llm_config_id: int | None,
) -> None:
    if llm_config_id is None:
        return
    await sync_to_async(_validate_llm_configuration_id)(llm_config_id)


async def _async_stream_codebase_agent(
    normalized_query: str,
    *,
    llm_config_id: int | None,
) -> AsyncIterator[str]:
    agent = await _build_codebase_agent_async(llm_config_id=llm_config_id)
    async for event in _astream_agent_output(agent, normalized_query):
        yield event


def _build_codebase_agent(*, llm_config_id: int | None = None) -> CodebaseAgent:
    env = LocalEnvironment(
        config=LocalEnvironmentConfig(cwd=str(settings.BASE_DIR)),
    )
    config = AgentConfig(
        step_limit=settings.CODEBASE_AGENT_STEP_LIMIT,
        cost_limit=settings.CODEBASE_AGENT_COST_LIMIT,
    )
    lm = None
    if llm_config_id is not None:
        _validate_llm_configuration_id(llm_config_id)
        llm_config = LlmConfiguration.objects.filter(pk=llm_config_id).first()
        assert llm_config is not None
        lm = build_dspy_lm_from_configuration(llm_config)
    return _default_agent_factory(env, config, lm=lm)


async def _build_codebase_agent_async(
    *,
    llm_config_id: int | None = None,
) -> CodebaseAgent:
    return await sync_to_async(_build_codebase_agent)(llm_config_id=llm_config_id)


def _default_agent_factory(
    env: LocalEnvironment,
    config: AgentConfig,
    *,
    lm: dspy.BaseLM | None = None,
) -> DefaultAgent:
    return DefaultAgent(env=env, config=config, lm=lm)


def _prediction_result(prediction: dspy.Prediction) -> str:
    result = getattr(prediction, "result", "")
    if not result:
        result = getattr(prediction, "process_result", "")
    return str(result).strip()


async def _astream_agent_output(agent: CodebaseAgent, task: str) -> AsyncIterator[str]:
    final_prediction: dspy.Prediction | None = None
    async for chunk in agent.stream(task):
        if isinstance(chunk, StreamResponse):
            yield _stream_event("token", content=chunk.chunk)
        elif isinstance(chunk, dspy.Prediction):
            final_prediction = chunk

    if final_prediction is None:
        raise RuntimeError("DSPy stream finished without returning a prediction.")

    agent.add_messages(agent._prediction_to_message(final_prediction))
    result = _prediction_result(final_prediction)
    if not result:
        raise ValueError("Codebase agent completed without a result.")
    yield _stream_event("result", content=result)


async def _consume_agent_stream(
    agent: CodebaseAgent,
    task: str,
    output_queue: queue.Queue[str | BaseException | None],
) -> None:
    final_prediction: dspy.Prediction | None = None
    async for chunk in agent.stream(task):
        if isinstance(chunk, StreamResponse):
            output_queue.put(_stream_event("token", content=chunk.chunk))
        elif isinstance(chunk, dspy.Prediction):
            final_prediction = chunk

    if final_prediction is None:
        raise RuntimeError("DSPy stream finished without returning a prediction.")

    agent.add_messages(agent._prediction_to_message(final_prediction))
    result = _prediction_result(final_prediction)
    if not result:
        raise ValueError("Codebase agent completed without a result.")
    output_queue.put(_stream_event("result", content=result))


def _stream_event(
    event_type: str,
    *,
    content: str | None = None,
    detail: str | None = None,
) -> str:
    event = {"type": event_type}
    if content is not None:
        event["content"] = content
    if detail is not None:
        event["detail"] = detail
    return f"{json.dumps(event)}\n"
