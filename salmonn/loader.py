"""Lock-driven SALMONN-13B loader (f_S).

Reads every component path + architecture hyper-parameter from
``env/MODEL_PINS.lock.json`` (nothing model-defining is hardcoded), sets the
pinned determinism flags/seeds, and returns a ready ``MelGuardSALMONN`` on GPU.
"""
from __future__ import annotations

import os
import random

from melguard.locks import model_pins, number_lock
from salmonn.download import ensure_components


def set_determinism(seed: int | None = None) -> int:
    """Pinned determinism (spec §6 / MODEL_PINS.reproducibility). Returns the seed used."""
    import numpy as np
    import torch

    pins = model_pins()
    repro = pins.get("reproducibility", {})
    if seed is None:
        seed = int(number_lock()["attack_seeds"][0])  # 1729
    for k, v in repro.get("env", {}).items():
        os.environ.setdefault(k, str(v))
    os.environ.setdefault("PYTHONHASHSEED", "0")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return seed


def load_salmonn(device: str = "cuda:0", quantization: str | None = None, verify: bool = True):
    """Build SALMONN-13B from the pinned components. `quantization` defaults to the lock (nf4)."""
    from salmonn.model import MelGuardSALMONN

    p = model_pins()["models"]["salmonn"]
    arch = p["architecture"]
    paths = ensure_components(verify=verify)

    quant = quantization or _quant_from_lock(p)
    model = MelGuardSALMONN(
        ckpt=paths["ckpt_file"],
        whisper_path=paths["whisper_dir"],
        beats_path=paths["beats_file"],
        vicuna_path=paths["vicuna_dir"],
        speech_qformer_token_num=arch["speech_qformer_token_num"],
        speech_qformer_layer=arch["speech_qformer_layer"],
        lora=arch["lora"],
        lora_alpha=arch["lora_alpha"],
        lora_rank=arch["lora_rank"],
        lora_dropout=arch["lora_dropout"],
        second_per_frame=arch["second_per_frame"],
        second_stride=arch["second_stride"],
        quantization=quant,
        device=device,
    )
    return model, {"paths": paths, "arch": arch, "quantization": quant,
                   "prompt_pattern": arch["prompt_pattern"]}


def _quant_from_lock(p: dict) -> str:
    q = p.get("quantization", "").lower()
    if "nf4" in q or "4-bit" in q:
        return "nf4"
    if "8-bit" in q or "int8" in q:
        return "int8"
    return "fp16"
