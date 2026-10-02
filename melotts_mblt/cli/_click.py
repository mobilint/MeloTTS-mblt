"""Run the vendored Click commands from argparse-based entry points with Click's own error reporting."""

from __future__ import annotations

import sys
from typing import Any, Sequence


def invoke_click(command: Any, args: Sequence[str], prog_name: str | None = None) -> int:
    """Invoke a Click ``command`` without letting it exit the interpreter, and return its exit status.

    ``standalone_mode=False`` keeps Click from calling ``sys.exit``, but it then raises ``ClickException`` (usage
    errors, bad parameters, missing arguments) and ``Abort`` instead of reporting them. Render those the way
    standalone mode would: Click's message on stderr and Click's exit code (2 for usage errors).

    Args:
        command: A Click command object.
        args: Arguments for the command.
        prog_name: Program name shown in usage and help output.

    Returns:
        The process exit status.
    """
    import click

    try:
        result = command.main(standalone_mode=False, prog_name=prog_name, args=list(args))
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.Abort:
        print("Aborted!", file=sys.stderr)
        return 1
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    # With standalone_mode=False, `--help` and `ctx.exit(code)` return the exit code instead of raising.
    return result if isinstance(result, int) else 0
