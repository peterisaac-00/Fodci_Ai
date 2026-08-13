"""Console entry point for the Backend Engineering Agent."""

from __future__ import annotations

import sys

from backend_ai.application import run_application
from backend_ai.core import InvalidProjectRootError


def main() -> int:
    """Start the application and enter its interactive session."""

    print("Backend Engineering Agent")
    try:
        run_application(output=sys.stdout)
    except KeyboardInterrupt:
        return 0
    except InvalidProjectRootError as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
