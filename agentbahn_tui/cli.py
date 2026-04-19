from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from django.core.management import execute_from_command_line


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "agentbahn.settings")
    arguments = list(argv) if argv is not None else sys.argv[1:]
    execute_from_command_line(["agentbahn-tui", "start_tui", *arguments])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
