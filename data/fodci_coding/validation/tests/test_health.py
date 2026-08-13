"""Validation corpus: health checks distinguish liveness from readiness."""

from dataclasses import dataclass

from api.validation import ListQuery, encode_error


@dataclass
class FakeDatabase:
    available: bool = True

    def execute(self, query: str) -> None:
        if not self.available:
            raise ConnectionError("database unavailable")
        assert query == "SELECT 1"


def test_list_query_rejects_unbounded_page_size() -> None:
    try:
        ListQuery.parse({"limit": "1000"})
    except ValueError as exc:
        assert "between 1 and 100" in str(exc)
    else:
        raise AssertionError("invalid page size was accepted")


def test_error_envelope_is_stable_json() -> None:
    assert encode_error("not_found", "resource does not exist") == (
        '{"error": {"code": "not_found", "detail": "resource does not exist"}}'
    )
