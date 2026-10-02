import json
import os

import torch
from huggingface_hub import hf_hub_download

from .utils import get_hparams_from_file

LANG_TO_HF_REPO_ID = {
    "EN_NEWEST": "mobilint/MeloTTS-English-v3",
    "KR": "mobilint/MeloTTS-Korean",
}

def load_or_download_config(locale, config_path=None, local_files_only=False):
    if config_path is None:
        language = locale.split("-")[0].upper()
        assert language in LANG_TO_HF_REPO_ID
        config_path = hf_hub_download(
            repo_id=LANG_TO_HF_REPO_ID[language],
            filename="config.json",
            local_files_only=local_files_only
        )
    return get_hparams_from_file(config_path)

def load_or_download_model(locale, device, ckpt_path=None, local_files_only=False):
    if ckpt_path is None:
        language = locale.split("-")[0].upper()
        assert language in LANG_TO_HF_REPO_ID
        ckpt_path = hf_hub_download(
            repo_id=LANG_TO_HF_REPO_ID[language],
            filename="checkpoint.pth",
            local_files_only=local_files_only
        )
    return torch.load(ckpt_path, map_location=device)


def resolve_local_mxq(repo_id, mxq_path):
    """Resolve ``mxq_path`` to a local file without network access.

    ``mblt_npu.MobilintNPUBackend`` downloads a missing MXQ from the Hub and has no local-only mode, so
    ``local_files_only=True`` callers resolve the file here first: an existing path is returned as-is, otherwise the
    Hugging Face cache is consulted with ``local_files_only=True`` (raising ``LocalEntryNotFoundError`` when absent).
    """
    if mxq_path and os.path.exists(mxq_path):
        return mxq_path
    return hf_hub_download(repo_id=repo_id, filename=mxq_path, local_files_only=True)


def resolve_local_bert_mxq(bert_model_id):
    """Return the cached MXQ path of a Mobilint BERT repository, or ``None`` if its config names none."""
    config_path = hf_hub_download(repo_id=bert_model_id, filename="config.json", local_files_only=True)
    with open(config_path, encoding="utf-8") as f:
        mxq_path = json.load(f).get("mxq_path")
    return resolve_local_mxq(bert_model_id, mxq_path) if mxq_path else None
