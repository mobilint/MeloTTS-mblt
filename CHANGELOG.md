# Changelog

## 0.0.0

### Added

- Initial standalone release, extracted from `mblt-model-zoo` 2.10.0 (`mblt_model_zoo.MeloTTS`, `origin/master`
  `083b5d6`). The `TTS` API, NPU-backed synthesizer, and English/Korean text processing are carried over under the
  `melotts_mblt` namespace (`from mblt_model_zoo.MeloTTS.api import TTS` → `from melotts_mblt import TTS`).
- `melotts-mblt` command with `tts` (the Click MeloTTS CLI), `ui` (the Gradio WebUI), and `download` (NLTK tagger and
  UniDic dictionary). These replace `mblt-model-zoo melo` / `melotts`, `mblt-model-zoo melo-ui`, and
  `mblt-melotts-download`.
- `import melotts_mblt` is lazy; `TTS` loads `torch` and `transformers` on first access.

### Changed

- The NPU backend comes from the required `mblt-npu-python>=0.1.0` dependency, and the Mobilint BERT prosody encoder
  from the required `transformers-mblt>=0.0.0` dependency. The text-processing and UI packages that were the Model Zoo
  `MeloTTS` extra are now required dependencies.
- Python examples pass `trust_remote_code=True`, which the Mobilint BERT encoder needs; the CLI and WebUI already did.
- Install hints and CLI help now point to `melotts-mblt`.

### Licensing

- The MeloTTS-derived code keeps MyShell.ai's MIT license (`melotts_mblt/LICENSE`, shipped in the wheel); Mobilint
  code is BSD-3-Clause. The project license is declared as `BSD-3-Clause AND MIT`.
