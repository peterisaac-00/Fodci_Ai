"""Console entry point for the Backend Engineering Agent."""

from __future__ import annotations

from backend_ai.application import start_application


def main() -> int:
    """Start the application and return a process status code."""

    print("Backend Engineering Agent")
    print("Starting application...")
    start_application()
    print("Application started successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
