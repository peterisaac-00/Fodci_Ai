"""Repository code keeps SQL and transaction boundaries explicit."""

from dataclasses import dataclass
from typing import Any, Protocol


class Connection(Protocol):
    def execute(self, query: str, params: tuple[Any, ...] = ()) -> Any: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


@dataclass(frozen=True)
class Project:
    project_id: str
    owner_id: str
    name: str


class ProjectRepository:
    def __init__(self, connection: Connection) -> None:
        self.connection = connection

    def find_by_id(self, project_id: str, owner_id: str) -> Project | None:
        row = self.connection.execute(
            "SELECT id, owner_id, name FROM projects WHERE id = %s AND owner_id = %s",
            (project_id, owner_id),
        ).fetchone()
        if row is None:
            return None
        return Project(project_id=row[0], owner_id=row[1], name=row[2])

    def create(self, project: Project) -> Project:
        try:
            self.connection.execute(
                "INSERT INTO projects (id, owner_id, name) VALUES (%s, %s, %s)",
                (project.project_id, project.owner_id, project.name),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return project

    def delete(self, project_id: str, owner_id: str) -> bool:
        result = self.connection.execute(
            "DELETE FROM projects WHERE id = %s AND owner_id = %s",
            (project_id, owner_id),
        )
        self.connection.commit()
        return result.rowcount == 1
