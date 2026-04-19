from __future__ import annotations

from ninja import Schema


class ProjectResponse(Schema):
    id: int
    name: str
    description: str
    date_created: str
    date_updated: str


ProjectListResponse = list[ProjectResponse]


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
