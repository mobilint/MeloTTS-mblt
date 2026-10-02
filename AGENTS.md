---
description: Guidance for coding agents working on the PyPI-distributed melotts-mblt package.
paths:
  - "**"
---

# melotts-mblt Agent Guide

This is the one guide for every coding agent. `CLAUDE.md` is a symlink to this file. For focused model, text
processing, CLI, and test work, also read `.claude/skills/melotts-mblt/SKILL.md`. For documentation work, read
`.claude/skills/melotts-mblt-readme/SKILL.md`. Each `.agents/skills/<name>` entry is a symlink to the matching `.claude`
skill, so edit the `.claude` copy only. User and system instructions take precedence over this guide.

## Mission

`melotts-mblt` is the standalone home of Mobilint's MeloTTS integration. It covers:

- the `TTS` Python API, the NPU-backed acoustic models, and text processing for English and Korean
- the `melotts-mblt` CLI: `tts`, `ui`, and `download`
- the test suite

`mblt-model-zoo` used to ship this code as `mblt_model_zoo.MeloTTS`. It depends on this package through its `MeloTTS`
extra and keeps only a forwarding facade and CLI bridges, the same way it does for transformers-mblt.

Ownership boundary:

- This package owns everything under `melotts_mblt/` and `tests/`. New MeloTTS features land here first.
- `mblt-npu-python` (`mblt_npu`) owns the NPU backend (`MobilintNPUBackend`). Do not vendor or fork backend code.
- `transformers-mblt` owns the Mobilint BERT prosody encoder (`mobilint-bert`), which `TTS` loads with
  `AutoModelForMaskedLM`. Do not copy Transformers integration code here.
- Do not import `mblt_model_zoo` from package code.

## Before Editing

- Run `git status --short` and preserve unrelated changes.
- Read `pyproject.toml`, `README.md`, `melotts_mblt/README.md`, the affected module, and the nearby tests before
  changing a public contract.
- Until the Model Zoo facade lands, `../mblt-model-zoo` at `origin/master` (`083b5d6`) is the behavioral reference. It
  holds the same code under `mblt_model_zoo/MeloTTS`. Before changing behavior, compare against that copy on purpose.

## Package and Dependency Contract

- The import namespace is `melotts_mblt`, and the distribution name is `melotts-mblt`. The version comes from
  `melotts_mblt.__version__`.
- `import melotts_mblt` is lazy and must not import `torch` or `transformers`. `TTS` is resolved on first access. Add
  new public names to `__all__` and `__getattr__` in `melotts_mblt/__init__.py`.
- The text-processing stack (`g2p_en`, `anyascii`, `jamo`, `g2pkk`, `unidic`, `python-mecab-ko`) and the UI stack
  (`soundfile`, `gradio`, `click`) are required dependencies. There are no extras.
- `melotts_mblt/text/cmudict.rep` and `text/cmudict_cache.pickle` are package data and must ship in the wheel.

## Licensing

- Mobilint code is BSD-3-Clause (`LICENSE`). The MeloTTS-derived code keeps MyShell.ai's MIT license
  (`melotts_mblt/LICENSE`), which must ship in the wheel. The project license is declared as `BSD-3-Clause AND MIT`.
- Keep the original-author credits and the citation in `melotts_mblt/README.md`. Do not remove or relicense
  upstream notices.

## NPU and Model Contract

- `TTS(language, ...)` downloads `mobilint/MeloTTS-English-v3` (`EN_NEWEST`) or `mobilint/MeloTTS-Korean` (`KR`) and
  runs the synthesizer MXQs through `mblt_npu.MobilintNPUBackend`. `dev_no`, `target_core`, `target_device`, and
  `encoder_mxq_path` / `decoder_mxq_path` override the config, and `target_device` is forwarded to both the
  synthesizer and the BERT encoder.
- The BERT encoder is a `mobilint/*` Hub repository with remote code, so `TTS` needs `trust_remote_code=True` (the CLI
  and WebUI pass it). Keep that documented in every Python example.
- Always `dispose()` a `TTS` when done; tests use `tests/pipe_teardown.py::pipe_fixture` for that.

## CLI Contract

- `melotts-mblt` (`melotts_mblt.cli:main`) provides `tts`, `ui`, and `download`. `tts` is dispatched before argparse so
  the Click command keeps owning its arguments and `--help`.
- Model Zoo bridges `run_tts`, `run_ui`, `run_download`, `add_tts_parser`, `add_ui_parser`, and `add_download_parser`
  for its `melo` / `melotts`, `melo-ui`, and `mblt-melotts-download` commands. Keep those names and signatures stable.

## Tests

- `tests/TEST.md` is the test guide. `tests/test_cli_main.py` and `tests/test_hparams.py` are hardware-free;
  `tests/test_melo.py::test_melo[...]` downloads each language model and synthesizes speech on the NPU.

## PyPI and Wheel Packaging

- Build from a clean tree with `python -m build`. The wheel must contain `melotts_mblt/py.typed`,
  `melotts_mblt/LICENSE`, both `cmudict` files, and `melotts_mblt/cli/main.py`, and must not contain `tests/`.
  `.github/workflows/publish.yml` checks this before publishing to TestPyPI and then PyPI.
- Record user-visible changes in `CHANGELOG.md` under a `## X.Y.Z` heading.

## Code Quality and Documentation

- Use four-space indentation, PEP 484 annotations, Google-style docstrings, and 120-character lines. Ruff covers
  `melotts_mblt/__init__.py`, `melotts_mblt/cli/`, and `tests/`. The extracted MeloTTS modules keep their upstream
  style and are excluded from Ruff for now.
- Install hints name `melotts-mblt`, not `mblt-model-zoo[MeloTTS]`.
- Keep `README.md`, `melotts_mblt/README.md`, CLI `--help` text, and this guide synchronized when a public API, CLI
  option, dependency, or ownership boundary changes.

## Git Safety

- Use Conventional Commit subjects under 50 characters.
- Do not revert, format, or regenerate unrelated files. Do not add model weights, `.mxq` files, generated audio, or
  caches unless explicitly requested.
