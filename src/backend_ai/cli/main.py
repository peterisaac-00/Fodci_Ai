"""Console entry point for the Backend Engineering Agent."""

from __future__ import annotations

import sys

from backend_ai.application import run_application


def main() -> int:
    """Start the application and enter its interactive session."""

    print("Backend Engineering Agent")
    run_application(output=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
