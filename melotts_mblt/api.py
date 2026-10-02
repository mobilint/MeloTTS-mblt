import math
import re
from typing import Optional

import numpy as np
import soundfile
import torch
from torch import nn
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForMaskedLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from . import utils
from .download_utils import (
    LANG_TO_HF_REPO_ID,
    load_or_download_config,
    load_or_download_model,
    resolve_local_bert_mxq,
    resolve_local_mxq,
)
from .models import MobilintSynthesizerTrn
from .split_utils import split_sentence
from .text.cleaner import clean_text


def _validate_speed(speed):
    """Return ``speed`` as a ``float`` if it is a finite real number greater than 0, else raise ``ValueError``.

    Any real scalar ``float()`` accepts is allowed (``int``, ``float``, NumPy scalars, 0-d tensors, ``Fraction``,
    ``Decimal``). Booleans and strings are rejected even though ``float()`` would convert them.
    """
    if isinstance(speed, (bool, np.bool_, str, bytes)):
        raise ValueError(f"speed must be a finite number greater than 0, got {speed!r}")
    try:
        value = float(speed)
    except (TypeError, ValueError):
        raise ValueError(f"speed must be a finite number greater than 0, got {speed!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"speed must be a finite number greater than 0, got {speed!r}")
    return value


class TTS(nn.Module):
    def __init__(self, 
                language,
                device='auto',
                config_path=None,
                ckpt_path=None,

                trust_remote_code: Optional[bool]=None,
                local_files_only: Optional[bool]=None,
                
                dev_no: Optional[int] = None,
                target_core: Optional[str] = None,
                encoder_mxq_path: Optional[str] = None,
                decoder_mxq_path: Optional[str] = None,
                target_device: Optional[str] = None,
        ):
        nn.Module.__init__(self)
        if device == "auto":
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            if torch.backends.mps.is_available():
                device = "mps"
        if "cuda" in device:
            assert torch.cuda.is_available()

        # config_path = 
        hps = load_or_download_config(language, config_path=config_path, local_files_only=local_files_only)
        
        if dev_no is not None:
            hps.model.dev_no = dev_no
        
        if target_core is not None:
            hps.model.target_core = target_core
        
        if encoder_mxq_path is not None:
            hps.model.encoder_mxq_path = encoder_mxq_path
        
        if decoder_mxq_path is not None:
            hps.model.decoder_mxq_path = decoder_mxq_path

        resolved_target_device = (
            target_device
            if target_device is not None
            else getattr(hps.model, "target_device", "aries-rb")
        )
        hps.model.target_device = resolved_target_device

        if local_files_only:
            # mblt_npu downloads a missing MXQ from the Hub; resolve both synthesizer MXQs from the cache first so
            # local_files_only never reaches the network (LocalEntryNotFoundError when they are not cached).
            repo_id = LANG_TO_HF_REPO_ID[language]
            hps.model.encoder_mxq_path = resolve_local_mxq(repo_id, hps.model.encoder_mxq_path)
            hps.model.decoder_mxq_path = resolve_local_mxq(repo_id, hps.model.decoder_mxq_path)

        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        # A failure while building the synthesizer is cleaned up by MobilintSynthesizerTrn itself; from here on the
        # NPU backends exist, so assign self.model before anything else can fail (including the device transfer).
        self.model = MobilintSynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=num_tones,
            num_languages=num_languages,
            name_or_path=LANG_TO_HF_REPO_ID[language],
            **hps.model,
        )

        try:
            model = self.model.to(device)
            model.eval()
            self.model = model
            self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
            self.hps = hps
            self.device = device

            # load state_dict
            checkpoint_dict = load_or_download_model(language, device, ckpt_path=ckpt_path, local_files_only=local_files_only)
            self.model.load_state_dict(checkpoint_dict['model'], strict=True)
        
            language = language.split('_')[0]
            self.language = 'ZH_MIX_EN' if language == 'ZH' else language # we support a ZH_MIX_EN model

            self.tokenizer = AutoTokenizer.from_pretrained(
                hps.model.bert_model_id,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )

            bert_kwargs = {}
            if local_files_only:
                # Same reason as the synthesizer MXQs: resolve the BERT MXQ from the cache before mblt_npu sees it.
                bert_mxq_path = resolve_local_bert_mxq(hps.model.bert_model_id)
                if bert_mxq_path is not None:
                    bert_kwargs["mxq_path"] = bert_mxq_path

            # Keep a reference to the loaded BERT (it owns an NPU backend) before the transfer, so dispose() can
            # still release it if .to(device) fails.
            self.bert = AutoModelForMaskedLM.from_pretrained(
                hps.model.bert_model_id,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,

                dev_no=hps.model.dev_no,
                target_cores=[hps.model.target_core],
                target_device=hps.model.target_device,
                **bert_kwargs,
            )
            self.bert = self.bert.to(device)
    
        except BaseException:
            # Checkpoint, tokenizer, or BERT loading failed after the synthesizer's NPU backends were created.
            self.dispose()
            raise

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.0):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language, quiet=False):
        texts = split_sentence(text, language_str=language)
        if not quiet:
            print(" > Text split to sentences.")
            print("\n".join(texts))
            print(" > ===========================")
        return texts
    
    def _bert_token_count(self, text, language):
        """Number of BERT tokens (special tokens included) the text produces after normalization."""
        if language in ["EN", "ZH_MIX_EN"]:
            text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        _, _, _, word2ph = clean_text(text, language, tokenizer=self.tokenizer)
        return len(word2ph)

    def _bert_max_tokens(self):
        config = getattr(getattr(self, "bert", None), "config", None)
        limit = getattr(config, "max_position_embeddings", None) or getattr(self.tokenizer, "model_max_length", None)
        # Tokenizers without a configured maximum report a huge sentinel value; fall back to BERT's usual 512.
        return int(limit) if limit and limit < 100_000 else 512

    def _fit_piece_to_bert(self, text, language, limit):
        """Split ``text`` until every piece fits BERT's position limit.

        Sentence splitting only breaks at punctuation, so unpunctuated input can exceed BERT's position embeddings,
        which are computed for the whole input before the MXQ runs (synthesizer chunking cannot help). Bisect at
        whitespace, or at the middle character for an unbroken run; each piece then goes through normalization,
        G2P, and BERT on its own, so ``word2ph`` / phoneme alignment is recomputed per piece instead of truncated.
        """
        if self._bert_token_count(text, language) <= limit:
            return [text]
        words = text.split()
        if len(words) > 1:
            middle = len(words) // 2
            left, right = " ".join(words[:middle]), " ".join(words[middle:])
        else:
            middle = len(text) // 2
            left, right = text[:middle], text[middle:]
        if not left.strip() or not right.strip():
            return [text]
        return self._fit_piece_to_bert(left, language, limit) + self._fit_piece_to_bert(right, language, limit)

    def fit_pieces_to_bert(self, texts, language):
        """Return ``texts`` with every piece split as needed to fit BERT's maximum input length."""
        if getattr(self.hps.data, "disable_bert", False):
            return list(texts)
        limit = self._bert_max_tokens()
        pieces = []
        for text in texts:
            pieces.extend(self._fit_piece_to_bert(text, language, limit))
        return pieces

    def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False):
        speed = _validate_speed(speed)
        language = self.language
        texts = self.fit_pieces_to_bert(self.split_sentences_into_pieces(text, language, quiet), language)
        audio_list = []
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)
        for t in tx:
            if language in ["EN", "ZH_MIX_EN"]:
                t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id, tokenizer=self.tokenizer, bert=self.bert)
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                del phones
                speakers = torch.LongTensor([speaker_id]).to(device)
                audio = (
                    self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        speakers,
                        tones,
                        lang_ids,
                        bert,
                        ja_bert,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=1.0 / speed,
                    )[0][0, 0]
                    .data.cpu()
                    .float()
                    .numpy()
                )
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                #
            audio_list.append(audio)
        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)
        
        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(
                    output_path, audio, self.hps.data.sampling_rate, format=format
                )
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)

    def launch(self):
        self.model.launch()

    def dispose(self):
        model = getattr(self, "model", None)
        if model is not None:
            model.dispose()
        # ``self.bert`` is a sibling NPU-backed module built via
        # ``AutoModelForMaskedLM.from_pretrained``; if the loaded class is a
        # Mobilint Bert (``MobilintBertForMaskedLM``) it owns its own NPU
        # backend and must be released alongside the synthesizer, otherwise
        # LPDDR stays pinned between TTS instances.
        bert_dispose = getattr(getattr(self, "bert", None), "dispose", None)
        if callable(bert_dispose):
            bert_dispose()
