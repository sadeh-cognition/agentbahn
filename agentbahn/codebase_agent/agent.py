import json
import traceback
import asyncio
from collections.abc import AsyncIterator, Callable
from loguru import logger
from pathlib import Path
from typing import Any
import platform

import dspy
from dspy.streaming import StreamListener
from dspy.streaming import StreamResponse
from pydantic import BaseModel

from agentbahn.llms.services import build_dspy_lm_from_configuration
from .environment import LocalEnvironment
from .utils import recursive_merge

TRAJECTORY_FORMAT_VERSION = "agentbahn-dspy-1.0"


class AgentConfig(BaseModel):
    step_limit: int = 10
    """Maximum number of steps the agent can take."""
    cost_limit: float
    """Stop agent after exceeding (!) this cost."""


class PlainCodebaseAgentSignature(dspy.Signature):
    task: str = dspy.InputField(desc="Please solve this task.")
    system_infrormation: str = dspy.InputField(
        desc="Information about the system, such as OS, version, and machine type, that can be useful for solving the task."
    )
    result: str = dspy.OutputField(
        desc="The result of the agent's actions. This should be a concise summary of the outcome of the executed command(s)."
    )


if platform.system() == "Darwin":
    CodebaseAgentSignature = PlainCodebaseAgentSignature.with_instructions("""You are a helpful assistant that can interact with a computer.

You can run exactly ONE bash code block with ONE command (or commands connected with && or ||) using the `execute_action` tool.

Failure to follow these rules will cause your actions to be rejected.

You can execute bash commands and edit files to implement the necessary changes.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your changes and any possible problems.

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust

## Important Rules

1. Run one action at a time.
2. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

## Formatting Actions

Here is an example of a correct action:

<example_action>

action to execute:
```
ls -la
```
</example_response>

## Useful command examples

### Create a new file:

```
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:

<important>
You are on MacOS. For all the below examples, you need to use `sed -i ''` instead of `sed -i`.
</important>

```
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:

```
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run

```
anything
```
Use the available `execute_action` tool to complete the user request.
""")
else:
    CodebaseAgentSignature = PlainCodebaseAgentSignature.with_instructions("""You are a helpful assistant that can interact with a computer.

You can run exactly ONE bash code block with ONE command (or commands connected with && or ||) using the `execute_action` tool.

Failure to follow these rules will cause your actions to be rejected.

You can execute bash commands and edit files to implement the necessary changes.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your changes and any possible problems.

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust

## Important Rules

1. Run one action at a time.
2. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

## Formatting Actions

Here is an example of a correct action:

<example_action>

action to execute:
```
ls -la
```
</example_response>

## Useful command examples

### Create a new file:

```
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:

```
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:

```
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run

```
anything
```
Use the available `execute_action` tool to complete the user request.
""")


class DefaultAgent:
    def __init__(
        self,
        env: LocalEnvironment,
        *,
        lm: dspy.BaseLM | None = None,
        config: AgentConfig,
    ):
        self.config = config
        self.messages: list[dict] = []
        self.env = env
        self.lm = lm or build_dspy_lm_from_configuration()
        self.extra_template_vars = {}
        self.cost = 0.0
        self.n_calls = 0
        self.agent = self._build_react_agent()

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            {"n_model_calls": self.n_calls, "model_cost": self.cost},
            self.extra_template_vars,
            kwargs,
        )

    def add_messages(self, *messages: dict) -> list[dict]:
        logger.debug(messages)
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            {
                "role": "exit",
                "content": str(e),
                "extra": {
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            }
        )

    def run(
        self,
        task: str,
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> dspy.Prediction:
        """Query the model and return model messages. Override to add hooks."""
        return asyncio.run(self.arun(task, on_token=on_token))

    async def arun(
        self,
        task: str,
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> dspy.Prediction:
        """Run the agent and optionally handle streamed output tokens."""
        final_prediction: dspy.Prediction | None = None
        async for chunk in self.stream(task):
            if isinstance(chunk, StreamResponse):
                if on_token:
                    on_token(chunk.chunk)
            elif isinstance(chunk, dspy.Prediction):
                final_prediction = chunk
        if final_prediction is None:
            raise RuntimeError("DSPy stream finished without returning a prediction.")
        return final_prediction

    async def stream(
        self, task: str
    ) -> AsyncIterator[StreamResponse | dspy.Prediction]:
        """Stream ReAct thought tokens and yield the final prediction."""
        logger.info(f"Agent starting run with task: {task}")
        lm = self.lm or build_dspy_lm_from_configuration()
        with dspy.context(lm=lm):
            stream_agent = self._build_streaming_agent()
            output_stream = stream_agent(
                task=task,
                system_infrormation=platform.uname()._asdict(),
            )
            async for chunk in output_stream:
                yield chunk
        logger.info("Agent finished run.")

    def _build_react_agent(self) -> dspy.ReAct:
        return dspy.ReAct(
            CodebaseAgentSignature,
            tools=[self.execute_action],
            max_iters=self.config.step_limit,
        )

    def _build_streaming_agent(self) -> Callable[..., Any]:
        return dspy.streamify(
            self.agent,
            stream_listeners=self._build_stream_listeners(),
        )

    def _build_stream_listeners(self) -> list[StreamListener]:
        return [
            StreamListener(
                signature_field_name="next_thought",
                allow_reuse=True,
            )
        ]

    def execute_action(self, action: str) -> dict[str, Any]:
        """Execute an action in the codebase environment and return the result."""
        logger.debug(f"Agent executing action:\n{action}")
        result = self.env.execute(action)
        logger.debug(f"Agent action result:\n{result}")
        return result

    def _prediction_to_message(self, prediction: dspy.Prediction) -> dict:
        submission = str(getattr(prediction, "process_result", "")).strip()
        trajectory = dict(getattr(prediction, "trajectory", {}) or {})
        content = submission or str(getattr(prediction, "reasoning", "")).strip()
        return {
            "role": "exit",
            "content": content,
            "extra": {
                "actions": self._actions_from_trajectory(trajectory),
                "exit_status": "success",
                "submission": submission,
                "trajectory": trajectory,
            },
        }

    def _actions_from_trajectory(self, trajectory: dict[str, Any]) -> list[str]:
        actions: list[str] = []
        for key, tool_name in trajectory.items():
            if not key.startswith("tool_name_") or tool_name != "execute_action":
                continue
            step_number = key.removeprefix("tool_name_")
            tool_args = trajectory.get(f"tool_args_{step_number}", {})
            if isinstance(tool_args, dict) and isinstance(tool_args.get("action"), str):
                actions.append(tool_args["action"])
        return actions

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "agent_runtime": TRAJECTORY_FORMAT_VERSION,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": TRAJECTORY_FORMAT_VERSION,
        }
        return recursive_merge(agent_data, self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data
