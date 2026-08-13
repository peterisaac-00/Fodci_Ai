"""Small backend example: a validated JSON REST endpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from flask import Flask, jsonify, request

app = Flask(__name__)


@dataclass(frozen=True, slots=True)
class User:
    user_id: int
    email: str
    active: bool = True


USERS: dict[int, User] = {}


def _json_error(message: str, status: int):
    return jsonify({"error": message}), status


def _parse_user_payload(payload: Any) -> tuple[str, bool] | tuple[None, None]:
    if not isinstance(payload, dict):
        return None, None
    email = payload.get("email")
    active = payload.get("active", True)
    if not isinstance(email, str) or "@" not in email:
        return None, None
    if not isinstance(active, bool):
        return None, None
    return email, active


@app.get("/health")
def health() -> tuple[Any, int] | Any:
    return jsonify({"status": "ok"})


@app.get("/users/<int:user_id>")
def get_user(user_id: int):
    user = USERS.get(user_id)
    if user is None:
        return _json_error("user not found", 404)
    return jsonify(asdict(user))


@app.post("/users")
def create_user():
    email, active = _parse_user_payload(request.get_json(silent=True))
    if email is None or active is None:
        return _json_error("email and active are required", 400)
    user_id = max(USERS, default=0) + 1
    user = User(user_id=user_id, email=email, active=active)
    USERS[user_id] = user
    return jsonify(asdict(user)), 201


@app.delete("/users/<int:user_id>")
def delete_user(user_id: int):
    if USERS.pop(user_id, None) is None:
        return _json_error("user not found", 404)
    return "", 204


@app.errorhandler(404)
def handle_not_found(_error):
    return _json_error("route not found", 404)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
