from django.apps import AppConfig


class AgentbahnTuiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "agentbahn_tui"

    def ready(self) -> None:
        from agentbahn.codebase_agent.observability import configure_dspy_mlflow

        configure_dspy_mlflow()
