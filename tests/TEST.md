# Test `melotts-mblt`

You can validate Mobilint's MeloTTS integration with [`pytest`](https://docs.pytest.org/en/stable/). The snippets below assume your virtual environment is already activated.

## Install Development Dependencies

Install the package plus the developer tooling required by the test suite:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e . --group dev
```

## Run All Tests

Run the whole suite. The `test_melo[...]` cases download each language model and synthesize speech on the NPU:

```bash
pytest tests
```

## Run a Single Language Case

Use `-k` to run just one of the languages, e.g., tests only Korean:

```bash
pytest tests/test_melo.py -k "KR"
```
