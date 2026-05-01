import djclick as click
from loguru import logger
import dspy
from agentbahn.codebase_agent.agent import DefaultAgent, AgentConfig
from agentbahn.codebase_agent.environment import (
    LocalEnvironment,
    LocalEnvironmentConfig,
)


@click.command()
@click.argument("task", type=str)
@click.option(
    "--step-limit",
    type=int,
    default=10,
    help="Maximum number of steps the agent can take.",
)
@click.option(
    "--cost-limit",
    type=float,
    default=3.0,
    help="Stop agent after exceeding this cost.",
)
def command(task: str, step_limit: int, cost_limit: float):
    """Run the DefaultAgent with the given task."""
    logger.info(f"Starting DefaultAgent with task: {task}")
    logger.debug(f"Configuration: step_limit={step_limit}, cost_limit={cost_limit}")

    config = AgentConfig(step_limit=step_limit, cost_limit=cost_limit)
    env_config = LocalEnvironmentConfig()
    env = LocalEnvironment(config=env_config)

    agent = DefaultAgent(env=env, config=config)

    try:
        prediction = agent.run(task)
        result = getattr(prediction, "result", None)
        if result:
            click.secho(f"\nResult:\n{result}", fg="green")
        else:
            click.secho(f"\nFinished: {prediction}", fg="green")

        logger.info("Agent execution completed successfully.")

    except Exception as e:
        logger.exception("Agent failed with an exception.")
        agent.handle_uncaught_exception(e)
        click.secho(f"Agent failed: {e}", fg="red")
