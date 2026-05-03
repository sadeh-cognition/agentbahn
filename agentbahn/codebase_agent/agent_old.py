import json
import traceback
from loguru import logger
from collections.abc import Callable
from pathlib import Path
from typing import Any
import platform

import dspy
from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from agentbahn.llms.services import build_dspy_lm_from_configuration
from .environment import LocalEnvironment
from .exceptions import InterruptAgentFlow, LimitsExceeded
from .utils import recursive_merge

TRAJECTORY_FORMAT_VERSION = "agentbahn-dspy-1.0"


class AgentConfig(BaseModel):
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float
    """Stop agent after exceeding (!) this cost."""


class PlainCodebaseAgentSignature(dspy.Signature):
    task: str = dspy.InputField(desc="Please solve this task.")
    system_infrormation: str = dspy.InputField(
        desc="Information about the system, such as OS, version, and machine type, that can be useful for solving the task."
    )
    thought: str = dspy.OutputField(
        desc="The reasoning process of the agent leading to the action."
    )
    command_to_execute: str = dspy.OutputField(
        desc="The action you want to be execute."
    )


if platform.system() == "Darwin":
    CodebaseAgentSignature = PlainCodebaseAgentSignature.with_instructions("""You are a helpful assistant that can interact with a computer.

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process.

Failure to follow these rules will cause your response to be rejected.

Please solve the given task.

You can execute bash commands and edit files to implement the necessary changes.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your changes and any possible problems.

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust
6. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
    Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

## Important Rules

1. Every response must contain exactly one action
2. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
    However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

## Formatting your response

Here is an example of a correct response:

<example_response>
thought: I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.

command_to_execute:
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
Use the available codebase environment tool to complete the user request.
""")
else:
    CodebaseAgentSignature = PlainCodebaseAgentSignature.with_instructions("""You are a helpful assistant that can interact with a computer.

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process.

Failure to follow these rules will cause your response to be rejected.

Please solve the given task.

You can execute bash commands and edit files to implement the necessary changes.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your changes and any possible problems.

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust
6. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
    Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

## Important Rules

1. Every response must contain exactly one action
2. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
    However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

## Formatting your response

Here is an example of a correct response:

<example_response>
thought: I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.

command_to_execute:
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
Use the available codebase environment tool to complete the user request.
""")


class DefaultAgent:
    def __init__(
        self,
        env: LocalEnvironment,
        *,
        lm: dspy.BaseLM | None = None,
        config_class: type = AgentConfig,
        **kwargs,
    ):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.env = env
        self.lm = lm or build_dspy_lm_from_configuration()
        self.extra_template_vars = {}
        self.cost = 0.0
        self.n_calls = 0
        self.system_prompt = ""
        self.instance_prompt = ""
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

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        while True:
            try:
                self.step()
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict]:
        """Run the ReAct agent and record the result."""
        return [self.query()]

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        lm = self.lm or build_dspy_lm_from_configuration()
        with dspy.context(lm=lm):
            prediction = self.agent(
                task=self.extra_template_vars.get("task", ""),
                system_infrormation=str(platform.uname()._asdict()),
            )
        message = self._prediction_to_message(prediction)
        self.cost += self._latest_lm_cost(lm)
        return message

    def _build_react_agent(self) -> dspy.ReAct:
        return dspy.ReAct(
            CodebaseAgentSignature,
            tools=[self.execute_action],
            max_iters=self.config.step_limit or 20,
        )

    def execute_action(self, action: str) -> Any:
        """Execute an action in the codebase environment and return the result."""
        return self.env.execute(action)

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        outputs = [
            self.execute_action(action)
            for action in message.get("extra", {}).get("actions", [])
        ]
        observations = [
            {
                "role": "user",
                "content": output,
                "extra": {"observation": output},
            }
            for output in outputs
        ]
        return self.add_messages(*observations)

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

    def _latest_lm_cost(self, lm: dspy.BaseLM) -> float:
        history = getattr(lm, "history", [])
        if not history:
            return 0.0
        cost = history[-1].get("cost") or 0.0
        return float(cost)

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
