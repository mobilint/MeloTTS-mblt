"""``melotts-mblt tts``: the Click-based MeloTTS text-to-speech CLI."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _require_tts_deps() -> None:
    try:
        import click  # noqa: F401
    except Exception as e:
        print(
            "Missing dependencies for the MeloTTS CLI.\n"
            "Install with: pip install -U melotts-mblt\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        raise SystemExit(2)


def run_tts(args: Sequence[str], prog_name: str = "melotts-mblt tts") -> int:
    """Run the Click TTS CLI with ``args`` and return its exit status.

    Args:
        args: Arguments for the Click command, for example ``["Hello", "out.wav", "--language", "EN_NEWEST"]``.
        prog_name: Program name shown in Click's usage and help output.

    Returns:
        The process exit status.
    """
    _require_tts_deps()

    from .. import main as melo_main
    from ._click import invoke_click

    return invoke_click(melo_main.main, args, prog_name=prog_name)


def _cmd_tts(args: argparse.Namespace) -> int:
    return run_tts(args.tts_args)


def add_tts_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str = "tts",
    aliases: Sequence[str] = (),
) -> argparse.ArgumentParser:
    """Register the ``tts`` subcommand.

    Arguments are forwarded to the Click CLI so ``--help`` shows Click's help. ``main()`` dispatches this command
    before argparse runs, so the parser here mainly documents it in ``melotts-mblt --help``.
    """
    parser = subparsers.add_parser(name, aliases=list(aliases), add_help=False, help="Synthesize speech (MeloTTS CLI)")
    parser.add_argument("tts_args", nargs=argparse.REMAINDER)
    parser.set_defaults(_handler=_cmd_tts)
    return parser
