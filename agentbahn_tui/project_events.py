from __future__ import annotations

from collections.abc import Callable

from agentbahn.projects.schemas import EventLogListResponse
from agentbahn.projects.schemas import EventLogResponse
from agentbahn.projects.schemas import FeatureListResponse
from agentbahn.projects.schemas import TaskListResponse
from agentbahn_tui.events import fetch_event_logs
from agentbahn_tui.features import fetch_features
from agentbahn_tui.tasks import fetch_tasks

PROJECT_EVENT_ENTITY_TYPE = "Project"
FEATURE_EVENT_ENTITY_TYPE = "Feature"
TASK_EVENT_ENTITY_TYPE = "Task"


def fetch_project_events(
    project_id: int,
    *,
    fetch_features_command: Callable[
        [int | None], FeatureListResponse
    ] = fetch_features,
    fetch_tasks_command: Callable[[int], TaskListResponse] = fetch_tasks,
    fetch_event_logs_command: Callable[
        [str, int], EventLogListResponse
    ] = fetch_event_logs,
) -> EventLogListResponse:
    related_entity_ids: list[tuple[str, int]] = [
        (PROJECT_EVENT_ENTITY_TYPE, project_id)
    ]
    related_entity_ids.extend(
        (FEATURE_EVENT_ENTITY_TYPE, feature.id)
        for feature in fetch_features_command(project_id)
    )
    related_entity_ids.extend(
        (TASK_EVENT_ENTITY_TYPE, task.id) for task in fetch_tasks_command(project_id)
    )

    deduplicated_events: dict[int, EventLogResponse] = {}
    for entity_type, entity_id in related_entity_ids:
        for event in fetch_event_logs_command(entity_type, entity_id):
            deduplicated_events[event.id] = event

    return sorted(
        deduplicated_events.values(), key=lambda event: event.id, reverse=True
    )
