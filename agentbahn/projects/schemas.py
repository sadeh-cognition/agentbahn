from __future__ import annotations

from ninja import Schema


class ProjectResponse(Schema):
    id: int
    name: str
    description: str
    date_created: str
    date_updated: str


ProjectListResponse = list[ProjectResponse]
