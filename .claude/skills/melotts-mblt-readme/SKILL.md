---
name: melotts-mblt-readme
description: >-
  Write and maintain melotts-mblt README documentation, the API reference, CLI examples, the test guide, changelog
  entries, licensing notices, and Model Zoo migration notes.
---

# melotts-mblt README Writing

## Documentation Ownership

- Root `README.md` stays short. It covers the package purpose, installation (with the one-time `melotts-mblt download`),
  a minimal `TTS` example, CLI highlights, the Model Zoo mapping, development setup, and the license split.
- `melotts_mblt/README.md` is the API reference. It owns language and speaker examples, the full CLI examples, the
  WebUI, and the original-author credits and citation.
- `tests/TEST.md` owns pytest commands. `CHANGELOG.md` records user-visible changes under `## X.Y.Z`.
- Present Model Zoo only as migration context: `mblt_model_zoo.MeloTTS.*` maps to `melotts_mblt.*`, `mblt-model-zoo melo`
  to `melotts-mblt tts`, `melo-ui` to `ui`, and `mblt-melotts-download` to `download`.

## Accuracy Rules

- Every Python example uses `from melotts_mblt import TTS` and passes `trust_remote_code=True`.
- CLI examples use `melotts-mblt tts|ui|download`, and every flag shown must exist in `--help`.
- Install hints use `pip install melotts-mblt`.
- Never drop the MIT notice, the upstream author credits, or the citation.

## Style and Validation

- Use ATX headings, one blank line between blocks, hyphen lists, concise paragraphs, and language-tagged code fences.
- When APIs, CLI options, dependencies, or licensing change, update the relevant README, `AGENTS.md`, and the
  `melotts-mblt` skill in the same change.
- For documentation-only updates, run `git diff --check` and verify relative links and anchors.
