"""Focused backend tests used as local training material."""

from __future__ import annotations

from auth_service import hash_password, require_session, verify_password


def test_password_round_trip() -> None:
    salt, digest = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", salt, digest)
    assert not verify_password("wrong password", salt, digest)


def test_expired_session_is_rejected() -> None:
    session = type("Session", (), {"token": "token", "expires_at": 10, "user_id": 7})()
    try:
        require_session(session, now=10)
    except PermissionError as error:
        assert "expired" in str(error)
    else:
        raise AssertionError("expired sessions must be rejected")


def test_invalid_payload_is_rejected() -> None:
    payload = {"email": "not-an-email", "active": "yes"}
    assert not isinstance(payload.get("email"), str) or "@" in payload["email"]
