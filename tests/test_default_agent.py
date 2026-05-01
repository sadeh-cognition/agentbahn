from __future__ import annotations

from pathlib import Path

import dspy

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
        raise AssertionError("The injected step module should provide the prediction.")


def test_default_agent_uses_dspy_react_agent_for_agent_run() -> None:
    env = FakeEnvironment()

    def react_agent(**kwargs) -> dspy.Prediction:
        assert "workspace: agentbahn" in kwargs["system_prompt"]
        assert kwargs["user_request"] == "Find health"
        return dspy.Prediction(
            process_result="Found the health endpoint.",
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

    agent = DefaultAgent(
        env,
        lm=FakeLm(),
        react_agent=react_agent,
        system_template="workspace: {{ workspace }}",
        instance_template="{{ task }}",
    )

    result = agent.run("Find health")

    assert result["actions"] == ["search health"]
    assert result["exit_status"] == "success"
    assert result["submission"] == "Found the health endpoint."
    assert result["trajectory"]["observation_0"] == "agentbahn/api.py: health"
    assert env.actions == []
    assert [message["role"] for message in agent.messages] == [
        "system",
        "user",
        "exit",
    ]


def test_default_agent_exposes_environment_execute_action_tool() -> None:
    env = FakeEnvironment()
    agent = DefaultAgent(
        env,
        lm=FakeLm(),
        system_template="workspace: {{ workspace }}",
        instance_template="{{ task }}",
    )

    result = agent.execute_action("search health")

    assert result == "observed search health"
    assert env.actions == ["search health"]


def test_default_agent_builds_dspy_react_with_environment_tool() -> None:
    env = FakeEnvironment()
    agent = DefaultAgent(
        env,
        lm=FakeLm(),
        system_template="workspace: {{ workspace }}",
        instance_template="{{ task }}",
        step_limit=3,
    )

    react_agent = agent._build_react_agent()

    assert isinstance(react_agent, dspy.ReAct)
    assert react_agent.max_iters == 3
    assert "execute_action" in react_agent.tools


def test_default_agent_serializes_dspy_trajectory(tmp_path: Path) -> None:
    env = FakeEnvironment()

    def react_agent(**kwargs) -> dspy.Prediction:
        return dspy.Prediction(
            process_result="Complete.",
            trajectory={"tool_name_0": "finish", "tool_args_0": {}},
        )

    output_path = tmp_path / "trajectory.json"
    agent = DefaultAgent(
        env,
        lm=FakeLm(),
        react_agent=react_agent,
        system_template="system",
        instance_template="{{ task }}",
        output_path=output_path,
    )

    agent.run("Task")
    serialized = agent.serialize()

    assert serialized["trajectory_format"] == "agentbahn-dspy-1.0"
    assert serialized["info"]["exit_status"] == "success"
    assert output_path.exists()
