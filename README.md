# melotts-mblt

<!-- markdownlint-disable MD033 -->
<div align="center">
<p>
<a href="https://www.mobilint.com/" target="_blank">
<img src="https://raw.githubusercontent.com/mobilint/.github/main/assets/Mobilint_Logo_Primary.png" alt="Mobilint Logo" width="60%">
</a>
</p>
</div>
<!-- markdownlint-enable MD033 -->

Run [MeloTTS](https://github.com/myshell-ai/MeloTTS) text-to-speech on Mobilint NPUs. `melotts-mblt` ships
pre-quantized MeloTTS synthesizers for English and Korean. It provides a Python API, a command line, and a Gradio
WebUI. The acoustic model runs on the NPU, and its Mobilint BERT prosody encoder loads through
[transformers-mblt](https://github.com/mobilint/transformers-mblt).

Models run on Mobilint [ARIES](https://www.mobilint.com/aries) and [REGULUS](https://www.mobilint.com/regulus)
boards. Supported target-device identifiers are `aries-rb`, `regulus-ra`, `regulus-rb`, `regulus-ra-usb`, and
`regulus-rb-usb`.

Version `0.0.0` is the first standalone release. It was extracted from `mblt-model-zoo` 2.11.0.

## Installation

[![PyPI - Version](https://img.shields.io/pypi/v/melotts-mblt?logo=pypi&logoColor=white)](https://pypi.org/project/melotts-mblt/)
[![PyPI Downloads](https://static.pepy.tech/badge/melotts-mblt?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://clickpy.clickhouse.com/dashboard/melotts-mblt)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/melotts-mblt?logo=python&logoColor=gold)](https://pypi.org/project/melotts-mblt/)

```bash
pip install melotts-mblt
# One-time download of the NLTK tagger (English G2P) and the UniDic dictionary
melotts-mblt download
```

NPU execution requires a supported Mobilint NPU driver and device. Python packaging cannot run post-install steps, so
run `melotts-mblt download` once after installing.

## Quick start

```python
from melotts_mblt import TTS

model = TTS(language="EN_NEWEST", device="cpu", trust_remote_code=True)
speaker_ids = model.hps.data.spk2id
model.tts_to_file("Did you ever hear a folk tale about a giant turtle?", speaker_ids["EN-Newest"], "en.wav", speed=1.0)
model.dispose()
```

`trust_remote_code=True` lets the Mobilint BERT prosody encoder load its Hub remote code; the `melotts-mblt` CLI and
WebUI pass it for you. Use `language="KR"` with speaker `"KR"` for Korean. Each language downloads its model from the Mobilint Hugging Face
organization: [`mobilint/MeloTTS-English-v3`](https://huggingface.co/mobilint/MeloTTS-English-v3) and
[`mobilint/MeloTTS-Korean`](https://huggingface.co/mobilint/MeloTTS-Korean).

These `TTS(...)` arguments select the NPU: `dev_no`, `target_core`, `target_device`, `encoder_mxq_path`, and
`decoder_mxq_path`. The [API reference](melotts_mblt/README.md) has more examples.

## Command line

The `melotts-mblt` command provides three subcommands:

- `tts` synthesizes speech with the MeloTTS CLI.
- `ui` launches the Gradio WebUI.
- `download` fetches the NLTK and UniDic resources.

```bash
melotts-mblt tts "Text to read" output.wav --language EN_NEWEST --speed 1.2
melotts-mblt tts "text-to-speech 안녕하세요" kr.wav --language KR
melotts-mblt tts file.txt out.wav --file
melotts-mblt ui --host 0.0.0.0 --port 7860
melotts-mblt download
```

`python -m melotts_mblt.cli` is equivalent to `melotts-mblt`. Run `melotts-mblt tts --help` for every TTS option.

## Using with mblt-model-zoo

This package replaces `mblt_model_zoo.MeloTTS`. The modules have the same contents, and only the import prefix and the
command names differ:

| mblt-model-zoo | melotts-mblt |
| --- | --- |
| `from mblt_model_zoo.MeloTTS.api import TTS` | `from melotts_mblt import TTS` |
| `mblt_model_zoo.MeloTTS.<module>` | `melotts_mblt.<module>` |
| `mblt-model-zoo melo ...` / `mblt-model-zoo melotts ...` | `melotts-mblt tts ...` |
| `mblt-model-zoo melo-ui` | `melotts-mblt ui` |
| `mblt-melotts-download` | `melotts-mblt download` |

## Development

Use [uv](https://docs.astral.sh/uv/) to manage the development environment:

```bash
git clone https://github.com/mobilint/MeloTTS-mblt.git
cd MeloTTS-mblt
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e . --group dev
pre-commit install
```

The [test guide](tests/TEST.md) explains the hardware-free and NPU test runs.

## Support and issues

For installation, model, or runtime support, visit the [Mobilint forum](https://discuss.mobilint.com/)
or contact [tech-support@mobilint.com](mailto:tech-support@mobilint.com). Report reproducible
package issues in the [MeloTTS-mblt issue tracker](https://github.com/mobilint/MeloTTS-mblt/issues).

## License

Mobilint's code is distributed under the [BSD 3-Clause License](LICENSE). The MeloTTS-derived code in `melotts_mblt`
remains under [MyShell.ai's MIT License](melotts_mblt/LICENSE), and that license ships in the wheel. MeloTTS was
created by Wenliang Zhao, Xumin Yu, and Zengyi Qin; see the [API reference](melotts_mblt/README.md#original-authors)
for the original authors and the citation.
