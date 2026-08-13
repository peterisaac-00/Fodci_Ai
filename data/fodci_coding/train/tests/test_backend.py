"""Tests should exercise validation, authorization, and failure boundaries."""

from dataclasses import dataclass

from api.routes import CreateUserRequest, create_user_route, get_user_route


@dataclass
class User:
    id: str
    email: str
    display_name: str


class FakeUsers:
    def __init__(self) -> None:
        self.created: list[User] = []

    def create(self, *, email: str, display_name: str) -> User:
        user = User(str(len(self.created) + 1), email, display_name)
        self.created.append(user)
        return user

    def get(self, user_id: str) -> User | None:
        return next((user for user in self.created if user.id == user_id), None)


def test_create_user_rejects_invalid_request_before_service_call() -> None:
    users = FakeUsers()
    status, body = create_user_route(CreateUserRequest("bad", ""), users)
    assert status == 400
    assert "email must contain @" in body["details"]
    assert users.created == []


def test_get_user_returns_not_found_for_unknown_id() -> None:
    status, body = get_user_route("missing", FakeUsers())
    assert status == 404
    assert body == {"error": "not_found"}


def test_route_maps_service_failure_to_safe_response() -> None:
    class FailingUsers:
        def create(self, **_: str) -> User:
            raise RuntimeError("database details must not cross the API boundary")

    status, body = create_user_route(CreateUserRequest("user@example.test", "A"), FailingUsers())
    assert status == 500
    assert body == {"error": "internal_error"}
