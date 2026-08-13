"""Backend authentication boundaries with explicit failure handling."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Session:
    user_id: int
    token: str
    expires_at: int


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    if len(password) < 12:
        raise ValueError("password must contain at least 12 characters")
    actual_salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        actual_salt,
        120_000,
    )
    return actual_salt.hex(), digest.hex()


def verify_password(password: str, salt_hex: str, expected_hash: str) -> bool:
    try:
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False
    _, actual_hash = hash_password(password, salt)
    return hmac.compare_digest(actual_hash, expected_hash)


def issue_session(user_id: int, now: int, lifetime_seconds: int = 3600) -> Session:
    if user_id <= 0:
        raise ValueError("user_id must be positive")
    if lifetime_seconds <= 0:
        raise ValueError("lifetime_seconds must be positive")
    return Session(
        user_id=user_id,
        token=secrets.token_urlsafe(32),
        expires_at=now + lifetime_seconds,
    )


def require_session(session: Session | None, now: int) -> int:
    if session is None or not session.token:
        raise PermissionError("authentication required")
    if now >= session.expires_at:
        raise PermissionError("session expired")
    return session.user_id
