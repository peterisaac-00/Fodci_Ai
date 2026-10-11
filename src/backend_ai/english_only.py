"""Fail-closed English-only policy for Fodci training inputs.

This policy applies to training material, not to the general filesystem/tooling
subsystems. It rejects Arabic-script text instead of silently transliterating or
filtering it, preserving dataset provenance and making violations observable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


ENGLISH_ONLY_POLICY_VERSION = "fodci-english-only-v1"

# Arabic, Arabic supplement, Arabic extended ranges, presentation forms, and
# Arabic mathematical alphabetic symbols. Arabic-Indic digits alone are not
# rejected; the policy targets Arabic-script language content.
_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
    (0x1EE00, 0x1EEFF),
)


class EnglishOnlyViolation(ValueError):
    """Raised when training content contains Arabic-script text."""


def contains_arabic(text: str) -> bool:
    """Return whether *text* contains a character from an Arabic Unicode block."""

    if not isinstance(text, str):
        raise TypeError("English-only validation requires text values")
    return any(any(start <= ord(char) <= end for start, end in _ARABIC_RANGES) for char in text)


def validate_english_text(text: str, *, field: str = "text") -> None:
    """Reject Arabic-script text with a stable, bounded diagnostic."""

    if not isinstance(text, str):
        raise EnglishOnlyViolation(f"{field} must be text")
    if contains_arabic(text):
        raise EnglishOnlyViolation(f"{field} contains Arabic-script text; English-only policy rejected it")


def validate_english_payload(value: Any, *, field: str = "payload", max_depth: int = 12) -> None:
    """Validate all strings nested in a JSON-like training payload."""

    if max_depth < 0:
        raise EnglishOnlyViolation(f"{field} exceeds maximum nesting depth")
    if value is None or isinstance(value, (bool, int, float)):
        return
    if isinstance(value, str):
        validate_english_text(value, field=field)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            validate_english_text(str(key), field=f"{field}.key")
            validate_english_payload(item, field=f"{field}.{key}", max_depth=max_depth - 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, item in enumerate(value):
            validate_english_payload(item, field=f"{field}[{index}]", max_depth=max_depth - 1)
        return
    raise EnglishOnlyViolation(f"{field} contains an unsupported value type")


__all__ = [
    "ENGLISH_ONLY_POLICY_VERSION",
    "EnglishOnlyViolation",
    "contains_arabic",
    "validate_english_payload",
    "validate_english_text",
]
