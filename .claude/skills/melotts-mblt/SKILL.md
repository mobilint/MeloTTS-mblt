---
name: melotts-mblt
description: >-
  Work on melotts-mblt, Mobilint's MeloTTS text-to-speech package for NPUs: the TTS API, NPU-backed synthesizer and
  BERT wiring, English/Korean text processing, the melotts-mblt CLI (tts, ui, download), packaging, and tests, while
  preserving the Model Zoo bridge and upstream MIT licensing.
---

# melotts-mblt

## Start Here

1. Read `AGENTS.md`.
2. Run `git status --short` before changing files.
3. Read `pyproject.toml`, `tests/TEST.md`, and `melotts_mblt/README.md`.
4. For behavior questions, compare against `../mblt-model-zoo` at `origin/master` (`mblt_model_zoo/MeloTTS`). Until
   the Model Zoo facade lands, it is the reference.

## Package Layout

- `melotts_mblt/api.py`: `TTS`, which loads the Hub config, the synthesizer, and the BERT prosody encoder.
- `melotts_mblt/models.py`: NPU-backed synthesizer modules built on `mblt_npu.MobilintNPUBackend`.
- `melotts_mblt/text/`: language cleaners, G2P, symbols, and the packaged `cmudict` data.
- `melotts_mblt/main.py` and `app.py`: the Click TTS command and the Gradio WebUI, wrapped by `melotts_mblt/cli/`.
- `melotts_mblt/cli/`: `main` (`melotts-mblt`), `tts`, `ui`, `download`.

## Preserve Contracts

- Keep `import melotts_mblt` lazy; `TTS` resolves through `__getattr__`.
- `TTS` needs `trust_remote_code=True` for the Mobilint BERT; keep it in every Python example.
- Keep `target_device` forwarding to both the synthesizer and BERT (see `tests/test_melo.py`).
- Keep the CLI bridge functions (`run_tts`, `run_ui`, `run_download`, `add_*_parser`) stable for Model Zoo.
- Ship `melotts_mblt/LICENSE` (MIT) and the `cmudict` files in the wheel; keep upstream credits in the README.

## Validate Proportionately

- Hardware-free: `pytest tests/test_cli_main.py tests/test_hparams.py`.
- NPU: `pytest tests/test_melo.py` (downloads each language model). Use `-k KR` or `-k EN_NEWEST` to narrow.
- Report unavailable hardware or downloads instead of weakening tests.
