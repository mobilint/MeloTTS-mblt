from __future__ import annotations

import argparse
import sys

from .download import add_download_parser
from .tts import add_tts_parser, run_tts
from .ui import add_ui_parser

TTS_COMMANDS = frozenset({"tts"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="melotts-mblt",
        description="MeloTTS text-to-speech on Mobilint NPUs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_package_version()}")
    commands_parser = parser.add_subparsers(help="melotts-mblt commands")

    add_tts_parser(commands_parser)
    add_ui_parser(commands_parser)
    add_download_parser(commands_parser)

    return parser


def _package_version() -> str:
    from .. import __version__

    return __version__


def main() -> int:
    # The Click TTS CLI owns its arguments (including `--help`), so dispatch it before argparse sees them.
    if len(sys.argv) > 1 and sys.argv[1] in TTS_COMMANDS:
        return run_tts(sys.argv[2:], prog_name=f"melotts-mblt {sys.argv[1]}")

    parser = build_parser()
    args = parser.parse_args()

    if hasattr(args, "_handler"):
        return args._handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
