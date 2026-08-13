"""HTTP routes keep transport concerns separate from application services."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CreateUserRequest:
    email: str
    display_name: str

    def validate(self) -> list[str]:
        errors: list[str] = []
        if "@" not in self.email:
            errors.append("email must contain @")
        if not self.display_name.strip():
            errors.append("display_name is required")
        return errors


def create_user_route(request: CreateUserRequest, user_service: Any) -> tuple[int, dict[str, Any]]:
    """Return an HTTP-shaped result; the service owns persistence and policy."""

    errors = request.validate()
    if errors:
        return 400, {"error": "invalid_request", "details": errors}
    try:
        user = user_service.create(email=request.email, display_name=request.display_name)
    except ValueError as exc:
        return 409, {"error": "user_conflict", "detail": str(exc)}
    except Exception:
        # A real application logs the exception with a request id and hides internals.
        return 500, {"error": "internal_error"}
    return 201, {"id": user.id, "email": user.email, "display_name": user.display_name}


def get_user_route(user_id: str, user_service: Any) -> tuple[int, dict[str, Any]]:
    if not user_id:
        return 400, {"error": "missing_user_id"}
    user = user_service.get(user_id)
    if user is None:
        return 404, {"error": "not_found"}
    return 200, {"id": user.id, "email": user.email, "display_name": user.display_name}
