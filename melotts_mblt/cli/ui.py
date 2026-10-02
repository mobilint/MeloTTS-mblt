"""``melotts-mblt ui``: launch the MeloTTS Gradio WebUI."""

from __future__ import annotations

import argparse
import sys


def _require_ui_deps() -> None:
    try:
        import gradio  # noqa: F401
    except Exception as e:
        print(
            "Missing dependencies for the MeloTTS WebUI.\n"
            "Install with: pip install -U melotts-mblt\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        raise SystemExit(2)


def run_ui(share: bool = False, host: str | None = None, port: int | None = None) -> int:
    """Launch the Gradio WebUI and return its exit status.

    Args:
        share: Expose a publicly accessible shared Gradio link.
        host: Server bind address, for example ``0.0.0.0``.
        port: Server port, for example ``7860``.

    Returns:
        The process exit status.
    """
    _require_ui_deps()

    from .. import app as melo_app

    click_args: list[str] = []
    if share:
        click_args.append("--share")
    if host is not None:
        click_args.extend(["--host", host])
    if port is not None:
        click_args.extend(["--port", str(port)])

    try:
        melo_app.main(standalone_mode=False, args=click_args)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    return run_ui(share=args.share, host=args.host, port=args.port)


def add_ui_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str = "ui",
) -> argparse.ArgumentParser:
    """Register the ``ui`` subcommand."""
    parser = subparsers.add_parser(name, help="Launch the MeloTTS WebUI (Gradio)")
    parser.add_argument(
        "--share",
        "-s",
        action="store_true",
        default=False,
        help="Expose a publicly-accessible shared Gradio link.",
    )
    parser.add_argument("--host", default=None, help="Server host / bind address (e.g., 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Server port (e.g., 7860)")
    parser.set_defaults(_handler=_cmd_ui)
    return parser
