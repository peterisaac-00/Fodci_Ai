"""Validation-only example: transport schemas should reject malformed input early."""

from dataclasses import dataclass
import json


@dataclass(frozen=True)
class ListQuery:
    limit: int = 20
    cursor: str | None = None

    @classmethod
    def parse(cls, raw: dict[str, str]) -> "ListQuery":
        try:
            limit = int(raw.get("limit", "20"))
        except ValueError as exc:
            raise ValueError("limit must be an integer") from exc
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        cursor = raw.get("cursor") or None
        return cls(limit=limit, cursor=cursor)


def encode_error(code: str, detail: str) -> str:
    """Serialize a stable JSON error envelope without exposing stack traces."""

    return json.dumps({"error": {"code": code, "detail": detail}}, sort_keys=True)
