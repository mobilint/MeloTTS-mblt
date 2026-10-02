"""Hardware-free regression tests for TTS input limits, cleanup, local-only loading, and decoder trimming."""

from __future__ import annotations

import math
import types

import numpy as np
import pytest
import torch

from melotts_mblt import api as melo_api
from melotts_mblt import download_utils
from melotts_mblt.api import TTS


def _bare_tts(max_tokens: int = 10) -> TTS:
    """A ``TTS`` without NPU state; token count = words + 2 special tokens, like a word-level BERT tokenizer."""
    tts = TTS.__new__(TTS)
    torch.nn.Module.__init__(tts)
    tts.hps = types.SimpleNamespace(data=types.SimpleNamespace(disable_bert=False))
    tts.tokenizer = types.SimpleNamespace(model_max_length=max_tokens)
    tts._bert_token_count = lambda text, language: len(text.split()) + 2
    return tts


def test_fit_pieces_to_bert_keeps_every_piece_within_the_limit() -> None:
    tts = _bare_tts(max_tokens=10)
    text = " ".join(f"w{i}" for i in range(37))  # 39 tokens, no punctuation

    pieces = tts.fit_pieces_to_bert([text, "short piece"], "KR")

    assert all(len(piece.split()) + 2 <= 10 for piece in pieces)
    assert " ".join(pieces[:-1]).split() == text.split()  # nothing truncated, order preserved
    assert pieces[-1] == "short piece"


def test_fit_pieces_to_bert_splits_unbroken_runs_by_character() -> None:
    tts = _bare_tts(max_tokens=10)
    tts._bert_token_count = lambda text, language: len(text) + 2  # one token per character
    run = "가" * 30

    pieces = tts.fit_pieces_to_bert([run], "KR")

    assert "".join(pieces) == run
    assert all(len(piece) + 2 <= 10 for piece in pieces)


@pytest.mark.parametrize("speed", [0, -1.0, math.nan, math.inf, True])
def test_tts_to_file_rejects_invalid_speed(speed: object) -> None:
    tts = _bare_tts()
    with pytest.raises(ValueError, match="speed"):
        tts.tts_to_file("hello", 0, speed=speed)


@pytest.mark.parametrize("speed", ["0", "-1", "nan", "inf"])
def test_cli_rejects_invalid_speed_before_loading(speed: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from melotts_mblt.cli.tts import run_tts

    monkeypatch.setattr(melo_api, "TTS", lambda **kwargs: pytest.fail("model must not load for an invalid speed"))
    assert run_tts(["hi", "out.wav", "--speed", speed]) == 2


class _FakeTTS:
    instances: list[_FakeTTS] = []

    def __init__(self, fail: bool = False, **kwargs: object) -> None:
        self.disposed = False
        self.fail = fail
        self.hps = types.SimpleNamespace(data=types.SimpleNamespace(spk2id={"EN-Newest": 0, "KR": 0}))
        _FakeTTS.instances.append(self)

    def tts_to_file(self, *args: object, **kwargs: object) -> None:
        if self.fail:
            raise RuntimeError("synthesis failed")

    def dispose(self) -> None:
        self.disposed = True


@pytest.mark.parametrize("fail", [False, True])
def test_cli_disposes_the_model_on_success_and_failure(fail: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from melotts_mblt.cli.tts import run_tts

    _FakeTTS.instances = []
    monkeypatch.setattr(melo_api, "TTS", lambda **kwargs: _FakeTTS(fail=fail, **kwargs))

    if fail:
        with pytest.raises(RuntimeError, match="synthesis failed"):
            run_tts(["hi", "out.wav"])
    else:
        assert run_tts(["hi", "out.wav"]) == 0
    assert [instance.disposed for instance in _FakeTTS.instances] == [True]


def _patch_tts_init(monkeypatch: pytest.MonkeyPatch, *, bert_fails: bool = False) -> dict[str, object]:
    """Replace config, checkpoint, synthesizer, tokenizer, BERT, and MXQ resolution with recording fakes."""
    seen: dict[str, object] = {"resolved": [], "synth": None, "bert_kwargs": None}

    class _ModelHParams(dict):
        """``hps.model`` stand-in: attribute access plus ``**`` unpacking, like ``HParams``."""

        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    hps = types.SimpleNamespace(
        model=_ModelHParams(
            dev_no=0,
            target_core="0:0",
            encoder_mxq_path="enc.mxq",
            decoder_mxq_path="dec.mxq",
            bert_model_id="mobilint/bert-kor-base",
            target_device="aries-rb",
        ),
        data=types.SimpleNamespace(filter_length=1024, hop_length=256, n_speakers=1),
        train=types.SimpleNamespace(segment_size=8192),
        num_languages=1,
        num_tones=1,
        symbols=["_"],
    )

    class _FakeSynth(torch.nn.Module):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()
            self.kwargs = kwargs
            self.disposed = False
            seen["synth"] = self

        def load_state_dict(self, *args: object, **kwargs: object) -> None:
            return None

        def dispose(self) -> None:
            self.disposed = True

    class _FakeBert:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            if bert_fails:
                raise OSError("BERT unavailable")
            seen["bert_kwargs"] = kwargs
            return types.SimpleNamespace(to=lambda device: types.SimpleNamespace(dispose=lambda: None))

    def _resolve(repo_id: str, path: str) -> str:
        seen["resolved"].append((repo_id, path))
        return f"/cache/{path}"

    monkeypatch.setattr(melo_api, "load_or_download_config", lambda *a, **k: hps)
    monkeypatch.setattr(melo_api, "load_or_download_model", lambda *a, **k: {"model": {}})
    monkeypatch.setattr(melo_api, "MobilintSynthesizerTrn", _FakeSynth)
    monkeypatch.setattr(melo_api, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *a, **k: object()))
    monkeypatch.setattr(melo_api, "AutoModelForMaskedLM", _FakeBert)
    monkeypatch.setattr(melo_api, "resolve_local_mxq", _resolve)
    monkeypatch.setattr(melo_api, "resolve_local_bert_mxq", lambda bert_id: f"/cache/{bert_id.split('/')[-1]}.mxq")
    return seen


def test_local_files_only_resolves_every_mxq_from_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch_tts_init(monkeypatch)

    TTS(language="KR", device="cpu", trust_remote_code=True, local_files_only=True)

    assert seen["resolved"] == [("mobilint/MeloTTS-Korean", "enc.mxq"), ("mobilint/MeloTTS-Korean", "dec.mxq")]
    assert seen["synth"].kwargs["encoder_mxq_path"] == "/cache/enc.mxq"
    assert seen["synth"].kwargs["decoder_mxq_path"] == "/cache/dec.mxq"
    assert seen["bert_kwargs"]["mxq_path"] == "/cache/bert-kor-base.mxq"


def test_default_loading_does_not_force_local_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch_tts_init(monkeypatch)

    TTS(language="KR", device="cpu", trust_remote_code=True)

    assert seen["resolved"] == []
    assert "mxq_path" not in seen["bert_kwargs"]


def test_failed_init_releases_the_synthesizer(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch_tts_init(monkeypatch, bert_fails=True)

    with pytest.raises(OSError, match="BERT unavailable"):
        TTS(language="KR", device="cpu", trust_remote_code=True)

    assert seen["synth"].disposed is True


def test_resolve_local_mxq_never_downloads(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    existing = tmp_path / "local.mxq"
    existing.write_bytes(b"")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(download_utils, "hf_hub_download", lambda **kwargs: calls.append(kwargs) or "/cache/x.mxq")

    assert download_utils.resolve_local_mxq("mobilint/MeloTTS-Korean", str(existing)) == str(existing)
    assert download_utils.resolve_local_mxq("mobilint/MeloTTS-Korean", "x.mxq") == "/cache/x.mxq"
    assert calls == [{"repo_id": "mobilint/MeloTTS-Korean", "filename": "x.mxq", "local_files_only": True}]


def test_decoder_trims_the_last_chunk_by_the_upsampling_factor() -> None:
    """Custom generators can have ``prod(upsample_rates) != upsample_initial_channel``; trim by the former."""
    from melotts_mblt.models import MobilintTransformerCouplingBlockAndGenerator

    factor, chunk, channels = 256, 128, 4
    decoder = MobilintTransformerCouplingBlockAndGenerator.__new__(MobilintTransformerCouplingBlockAndGenerator)
    torch.nn.Module.__init__(decoder)
    decoder.half_channels = channels // 2
    decoder.upsample_factor = factor
    decoder.allowed_chunks = [chunk]
    decoder.npu_backend = types.SimpleNamespace(
        mxq_model=types.SimpleNamespace(infer=lambda inputs: [np.zeros((1, chunk * factor, 1), dtype=np.float32)])
    )

    _, audio = decoder(torch.zeros(1, channels, 64))

    assert audio.shape[-1] == 64 * factor
