"""``melotts-mblt download``: fetch the language resources MeloTTS needs at runtime.

English G2P needs NLTK's ``averaged_perceptron_tagger_eng`` data, and Japanese text processing needs the UniDic
dictionary. Python packaging cannot run post-install steps, so this command downloads both.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run_download() -> int:
    """Download the NLTK tagger and the UniDic dictionary, returning the process exit status."""
    try:
        import nltk

        downloaded = nltk.download("averaged_perceptron_tagger_eng")
    except Exception as e:  # pragma: no cover
        print(
            "Could not download the NLTK tagger. Reinstall the package first:\n"
            "  pip install -U melotts-mblt\n"
            "\n"
            "Then run python code:\n"
            "  import nltk\n"
            "  nltk.download('averaged_perceptron_tagger_eng')\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        return 1
    # nltk.download() reports many failures (network, permissions) by returning False instead of raising.
    if not downloaded:
        print(
            "Downloading the NLTK tagger 'averaged_perceptron_tagger_eng' failed; see the NLTK messages above.\n"
            "English text-to-speech needs it. Retry with network access, or run in Python:\n"
            "  import nltk\n"
            "  nltk.download('averaged_perceptron_tagger_eng')",
            file=sys.stderr,
        )
        return 1

    try:
        import unidic  # noqa: F401
    except Exception as e:  # pragma: no cover
        print(
            "unidic is not installed. Reinstall the package first:\n"
            "  pip install -U melotts-mblt\n"
            "\n"
            "Then run:\n"
            "  python -m unidic download\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        return 1

    return subprocess.call([sys.executable, "-m", "unidic", "download"])


def _cmd_download(args: argparse.Namespace) -> int:
    return run_download()


def add_download_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str = "download",
) -> argparse.ArgumentParser:
    """Register the ``download`` subcommand."""
    parser = subparsers.add_parser(name, help="Download the NLTK tagger and UniDic dictionary used by MeloTTS")
    parser.set_defaults(_handler=_cmd_download)
    return parser


def main() -> int:
    """Entry point kept for ``python -m melotts_mblt.cli.download``."""
    return run_download()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
