import mlflow as mlflow_module

_mlflow_autolog_initialized = False


def configure_dspy_mlflow() -> bool:
    global _mlflow_autolog_initialized

    tracking_uri = "http://127.0.0.1:5000"

    if _mlflow_autolog_initialized:
        return True

    mlflow_module.set_tracking_uri(tracking_uri)
    mlflow_module.set_experiment("AgentBahn")
    mlflow_module.dspy.autolog()  # type: ignore
    _mlflow_autolog_initialized = True
    return True


def reset_dspy_mlflow_state() -> None:
    global _mlflow_autolog_initialized
    _mlflow_autolog_initialized = False
