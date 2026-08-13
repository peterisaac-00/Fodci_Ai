"""Command recognition without command execution."""

from __future__ import annotations

from dataclasses import dataclass
import re

_COMMAND_NAME = re.compile(r"[^\s]+")


@dataclass(frozen=True, slots=True)
class Command:
    """A recognized command and its unmodified argument content."""

    name: str
    arguments: str
    raw: str


@dataclass(frozen=True, slots=True)
class ParsedInput:
    """Result of classifying one received line."""

    text: str
    command: Command | None = None

    @property
    def is_command(self) -> bool:
        """Return whether the input begins with a valid command syntax."""

        return self.command is not None


class CommandParser:
    """Recognize commands only when ``/`` is the first character."""

    def parse(self, text: str) -> ParsedInput:
        """Classify text while preserving normal input exactly.

        Leading whitespace prevents command recognition by design. This makes
        the boundary explicit: a command must begin with ``/`` at position 0.
        Command names are normalized with ``casefold``; argument content is
        preserved after the delimiter whitespace.
        """

        if not text.startswith("/"):
            return ParsedInput(text=text)

        match = _COMMAND_NAME.match(text[1:])
        if match is None:
            return ParsedInput(text=text)

        name_end = 1 + match.end()
        arguments = text[name_end:].lstrip(" \t")
        command = Command(
            name=match.group().casefold(),
            arguments=arguments,
            raw=text,
        )
        return ParsedInput(text=text, command=command)
