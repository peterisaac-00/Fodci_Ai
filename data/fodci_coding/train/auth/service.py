"""Authentication service boundaries for a small HTTP application."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import scrypt
import hmac
import secrets
from typing import Protocol


class UserStore(Protocol):
    def find_by_email(self, email: str) -> "UserRecord | None": ...

    def find_by_id(self, user_id: str) -> "UserRecord | None": ...


@dataclass(frozen=True)
class UserRecord:
    user_id: str
    email: str
    password_digest: bytes
    password_salt: bytes
    roles: tuple[str, ...]


@dataclass(frozen=True)
class Session:
    user_id: str
    expires_at: datetime
    token: str


def hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    """Use a per-user random salt; never store a plaintext password."""

    if len(password) < 12:
        raise ValueError("password must contain at least twelve characters")
    chosen_salt = salt or secrets.token_bytes(16)
    digest = scrypt(password.encode("utf-8"), salt=chosen_salt, n=2**14, r=8, p=1)
    return digest, chosen_salt


def verify_password(password: str, record: UserRecord) -> bool:
    candidate, _ = hash_password(password, record.password_salt)
    return hmac.compare_digest(candidate, record.password_digest)


def authenticate(email: str, password: str, store: UserStore) -> Session | None:
    record = store.find_by_email(email)
    if record is None or not verify_password(password, record):
        return None
    return Session(
        user_id=record.user_id,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        token=secrets.token_urlsafe(32),
    )


def authorize(record: UserRecord | None, required_role: str) -> bool:
    return record is not None and required_role in record.roles
