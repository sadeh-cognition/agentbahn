from __future__ import annotations

import httpx

from agentbahn_tui.tasks import fetch_tasks


def test_fetch_tasks_filters_by_project_id(settings) -> None:
    settings.PROJECTBAHN_API_BASE_URL = "http://testserver"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tasks"
        assert request.url.params["project_id"] == "7"
        return httpx.Response(
            200,
            json=[
                {
                    "id": 11,
                    "project_id": 7,
                    "project_name": "Roadmap",
                    "feature_id": 3,
                    "feature_name": "CLI",
                    "user_id": 5,
                    "user_username": "alice",
                    "title": "List tasks",
                    "description": "Show tasks for a project",
                    "status": "todo",
                    "date_created": "2026-04-19T10:00:00Z",
                    "date_updated": "2026-04-19T10:30:00Z",
                }
            ],
        )

    tasks = fetch_tasks(7, transport=httpx.MockTransport(handler))

    assert len(tasks) == 1
    assert tasks[0].id == 11
    assert tasks[0].title == "List tasks"
