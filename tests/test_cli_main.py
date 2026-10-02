"""Hardware-free tests for the ``melotts-mblt`` entry point and package facade."""

from __future__ import annotations

import importlib
import subprocess
import sys
import types

import pytest

# ``melotts_mblt.cli`` re-exports the ``main`` function, which shadows the ``main`` module attribute.
cli_main = importlib.import_module("melotts_mblt.cli.main")


def test_build_parser_exposes_subcommands() -> None:
    parser = cli_main.build_parser()
    assert parser.prog == "melotts-mblt"
    ui_args = parser.parse_args(["ui", "--share", "--host", "0.0.0.0", "--port", "7860"])
    assert (ui_args.share, ui_args.host, ui_args.port) == (True, "0.0.0.0", 7860)
    # Short flags kept from the upstream `melo-ui` command (`-s` for --share, `-p` for --port).
    short_args = parser.parse_args(["ui", "-s", "-p", "7861"])
    assert (short_args.share, short_args.port) == (True, 7861)
    assert hasattr(parser.parse_args(["download"]), "_handler")


def test_tts_is_dispatched_before_argparse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Click owns the ``tts`` arguments, including ``--help`` and its own options."""
    seen: dict[str, object] = {}

    class _FakeClickCommand:
        def main(self, *, standalone_mode: bool, prog_name: str, args: list[str]) -> None:
            seen.update(standalone_mode=standalone_mode, prog_name=prog_name, args=args)

    monkeypatch.setattr("melotts_mblt.main.main", _FakeClickCommand())
    monkeypatch.setattr(cli_main, "build_parser", lambda: pytest.fail("argparse must not parse tts arguments"))
    monkeypatch.setattr(sys, "argv", ["melotts-mblt", "tts", "Hello", "out.wav", "--language", "KR"])

    assert cli_main.main() == 0
    assert seen == {
        "standalone_mode": False,
        "prog_name": "melotts-mblt tts",
        "args": ["Hello", "out.wav", "--language", "KR"],
    }


def test_run_tts_propagates_click_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    from melotts_mblt.cli.tts import run_tts

    class _ExitingCommand:
        def main(self, **kwargs: object) -> None:
            raise SystemExit(3)

    monkeypatch.setattr("melotts_mblt.main.main", _ExitingCommand())
    assert run_tts(["--help"]) == 3


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--bad-option"], "No such option"),
        ([], "Missing argument"),
        (["hi", "out.wav", "--language", "XX"], "Invalid value"),
    ],
)
def test_tts_usage_errors_exit_2_without_traceback(
    args: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Click usage errors are reported like standalone Click: usage on stderr and exit status 2."""
    from melotts_mblt.cli.tts import run_tts

    assert run_tts(args) == 2
    err = capsys.readouterr().err
    assert "Usage: melotts-mblt tts" in err
    assert message in err
    assert "Traceback" not in err


def test_tts_help_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    from melotts_mblt.cli.tts import run_tts

    assert run_tts(["--help"]) == 0
    assert "--language" in capsys.readouterr().out


def test_ui_builds_click_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    from melotts_mblt.cli.ui import run_ui

    calls: list[list[str]] = []
    fake_app = types.ModuleType("melotts_mblt.app")
    fake_app.main = types.SimpleNamespace(  # type: ignore[attr-defined]
        main=lambda *, standalone_mode, prog_name, args: calls.append(args)
    )
    monkeypatch.setitem(sys.modules, "melotts_mblt.app", fake_app)
    monkeypatch.setattr("melotts_mblt.app", fake_app, raising=False)

    assert run_ui(share=True, host="0.0.0.0", port=7860) == 0
    assert run_ui() == 0
    assert calls == [["--share", "--host", "0.0.0.0", "--port", "7860"], []]


def test_download_fetches_nltk_tagger_then_unidic(monkeypatch: pytest.MonkeyPatch) -> None:
    from melotts_mblt.cli import download

    steps: list[object] = []
    fake_nltk = types.ModuleType("nltk")
    fake_nltk.download = lambda resource: steps.append(("nltk", resource)) or True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nltk", fake_nltk)
    monkeypatch.setitem(sys.modules, "unidic", types.ModuleType("unidic"))
    monkeypatch.setattr(download.subprocess, "call", lambda argv: steps.append(("call", argv[1:])) or 0)

    assert download.run_download() == 0
    assert steps == [("nltk", "averaged_perceptron_tagger_eng"), ("call", ["-m", "unidic", "download"])]


def test_download_stops_when_nltk_reports_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``nltk.download`` returns False on many failures; the command must fail instead of continuing."""
    from melotts_mblt.cli import download

    fake_nltk = types.ModuleType("nltk")
    fake_nltk.download = lambda resource: False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nltk", fake_nltk)
    monkeypatch.setattr(download.subprocess, "call", lambda argv: pytest.fail("UniDic must not run after a failure"))

    assert download.run_download() == 1
    assert "averaged_perceptron_tagger_eng" in capsys.readouterr().err


def test_no_command_prints_help(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["melotts-mblt"])
    assert cli_main.main() == 1
    assert "usage: melotts-mblt" in capsys.readouterr().out


def test_module_entry_point_and_lazy_package() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, melotts_mblt; assert 'torch' not in sys.modules, 'eager torch import';"
            "from melotts_mblt import TTS; print(TTS.__module__)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "melotts_mblt.api"

    help_result = subprocess.run(
        [sys.executable, "-m", "melotts_mblt.cli", "--help"], capture_output=True, text=True, check=False
    )
    assert help_result.returncode == 0
    assert "melotts-mblt" in help_result.stdout
