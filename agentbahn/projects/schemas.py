from __future__ import annotations

from typing import Any

from ninja import Schema


class ProjectResponse(Schema):
    id: int
    name: str
    description: str
    date_created: str
    date_updated: str


ProjectListResponse = list[ProjectResponse]


class FeatureResponse(Schema):
    id: int
    project_id: int
    parent_feature_id: int | None = None
    name: str
    description: str
    date_created: str
    date_updated: str


FeatureListResponse = list[FeatureResponse]


class TaskResponse(Schema):
    id: int
    project_id: int
    project_name: str
    feature_id: int
    feature_name: str
    user_id: int
    user_username: str
    title: str
    description: str
    status: str
    date_created: str
    date_updated: str


TaskListResponse = list[TaskResponse]


class EventLogResponse(Schema):
    id: int
    entity_type: str
    entity_id: int
    event_type: str
    event_details: dict[str, Any]


EventLogListResponse = list[EventLogResponse]


class EventLogPageResponse(Schema):
    items: EventLogListResponse
    total: int
    page: int
    page_size: int
