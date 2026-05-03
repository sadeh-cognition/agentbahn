from __future__ import annotations

from pathlib import Path

import dspy

from agentbahn.codebase_agent.agent import AgentConfig
from agentbahn.codebase_agent.agent import DefaultAgent


class FakeEnvironment:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def get_template_vars(self) -> dict:
        return {"workspace": "agentbahn"}

    def execute(self, action: str) -> str:
        self.actions.append(action)
        return f"observed {action}"

    def serialize(self) -> dict:
        return {"environment": {"actions": self.actions}}


class FakeLm(dspy.BaseLM):
    def __init__(self) -> None:
        super().__init__(model="test/model")

    def forward(self, prompt=None, messages=None, **kwargs):
        raise AssertionError("The injected stream should provide the prediction.")


class StreamingDefaultAgent(DefaultAgent):
    async def stream(self, task: str):
        assert task == "Find health"
        yield dspy.streaming.StreamResponse(
            predict_name="react",
            signature_field_name="next_thought",
            chunk="Search",
            is_last_chunk=False,
        )
        yield dspy.streaming.StreamResponse(
            predict_name="react",
            signature_field_name="next_thought",
            chunk=" health",
            is_last_chunk=True,
        )
        yield dspy.Prediction(
            result="Found the health endpoint.",
            trajectory={
                "thought_0": "Search for health.",
                "tool_name_0": "execute_action",
                "tool_args_0": {"action": "search health"},
                "observation_0": "agentbahn/api.py: health",
                "tool_name_1": "finish",
                "tool_args_1": {},
                "observation_1": "Completed.",
            },
        )


def test_default_agent_run_streams_react_next_thought_tokens() -> None:
    env = FakeEnvironment()
    agent = StreamingDefaultAgent(
        env,
        lm=FakeLm(),
        config=AgentConfig(step_limit=3, cost_limit=0.25),
    )
    streamed_tokens: list[str] = []

    prediction = agent.run("Find health", on_token=streamed_tokens.append)

    assert streamed_tokens == ["Search", " health"]
    assert prediction.result == "Found the health endpoint."
    assert prediction.trajectory["observation_0"] == "agentbahn/api.py: health"
    assert env.actions == []


def test_default_agent_exposes_environment_execute_action_tool() -> None:
    env = FakeEnvironment()
    agent = DefaultAgent(env, lm=FakeLm(), config=AgentConfig(cost_limit=0.25))

    result = agent.execute_action("search health")

    assert result == "observed search health"
    assert env.actions == ["search health"]


def test_default_agent_builds_dspy_react_with_environment_tool() -> None:
    env = FakeEnvironment()
    agent = DefaultAgent(
        env, lm=FakeLm(), config=AgentConfig(step_limit=3, cost_limit=0.25)
    )

    react_agent = agent._build_react_agent()

    assert isinstance(react_agent, dspy.ReAct)
    assert react_agent.max_iters == 3
    assert "execute_action" in react_agent.tools


def test_default_agent_reuses_next_thought_stream_listener() -> None:
    env = FakeEnvironment()
    agent = DefaultAgent(env, lm=FakeLm(), config=AgentConfig(cost_limit=0.25))

    listeners = agent._build_stream_listeners()

    assert len(listeners) == 1
    assert listeners[0].signature_field_name == "next_thought"
    assert listeners[0].allow_reuse is True


def test_default_agent_serializes_dspy_trajectory(tmp_path: Path) -> None:
    env = FakeEnvironment()
    agent = StreamingDefaultAgent(
        env,
        lm=FakeLm(),
        config=AgentConfig(cost_limit=0.25),
    )

    prediction = agent.run("Find health")
    agent.add_messages(agent._prediction_to_message(prediction))
    output_path = tmp_path / "trajectory.json"
    serialized = agent.save(output_path)

    assert serialized["trajectory_format"] == "agentbahn-dspy-1.0"
    assert serialized["info"]["exit_status"] == "success"
    assert output_path.exists()
