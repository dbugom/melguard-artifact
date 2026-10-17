"""Phase 5, step 1 gate: forward-equivalence + gradient-to-waveform check for the
differentiable front-end (``attack/frontend.py``) against the deployed numpy
front-end in ``salmonn/model.py``.

PASS iff the torch front-end reproduces the deployed speech embeddings within
fp16 noise AND a finite, non-zero gradient reaches the waveform delta.

    python -m attack.equiv_check [--clip PATH] [--out out/phase5/frontend_equiv.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import soundfile as sf

from attack.frontend import forward_equivalence, grad_reaches_waveform
from salmonn.loader import load_salmonn, set_determinism

# PASS tolerances. The mel is fp32 (numpy-vs-torch STFT/FFT rounding); the
# embeddings pass through the fp16 Whisper+Q-Former stack, so parity is judged
# primarily on per-token cosine similarity.
TOL = {
    "mel_max_abs_diff": 5e-2,
    "embed_cosine_min": 0.999,
    "embed_rel_mean_diff": 5e-2,
}


def _find_clip() -> str:
    for sub in ("clips/commonvoice18", "clips/librispeech", "clips/fleurs"):
        for ext in ("flac", "wav"):
            hits = sorted(glob.glob(os.path.join("/workspace/data", sub, "**", f"*.{ext}"), recursive=True))
            if hits:
                return hits[0]
    raise FileNotFoundError("no sample clip found under /workspace/data/clips")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=None)
    ap.add_argument("--out", default="out/phase5/frontend_equiv.json")
    args = ap.parse_args()

    seed = set_determinism()
    clip = args.clip or _find_clip()
    wav, sr = sf.read(clip)
    if wav.ndim == 2:
        wav = wav[:, 0]
    wav = wav.astype(np.float32)
    assert sr == 16000, f"expected 16 kHz clip, got {sr}"

    print(f"[equiv] seed={seed} clip={clip} dur={len(wav) / sr:.2f}s")
    print("[equiv] loading SALMONN (f_S) ...")
    model, meta = load_salmonn(verify=False)

    fwd = forward_equivalence(model, wav)
    grad = grad_reaches_waveform(model, wav, target_len=16000)

    checks = {
        "mel_parity": fwd["mel_max_abs_diff"] <= TOL["mel_max_abs_diff"],
        "embed_cosine": fwd["embed_cosine_min"] >= TOL["embed_cosine_min"],
        "embed_rel": fwd["embed_rel_mean_diff"] <= TOL["embed_rel_mean_diff"],
        "grad_finite": grad["grad_is_finite"],
        "grad_nonzero": grad["grad_abs_sum"] > 0.0,
    }
    passed = all(checks.values())

    result = {
        "phase": 5,
        "step": "1_differentiable_frontend",
        "seed": seed,
        "clip": clip,
        "clip_seconds": len(wav) / sr,
        "quantization": meta["quantization"],
        "tolerances": TOL,
        "forward_equivalence": fwd,
        "grad_reaches_waveform": grad,
        "checks": checks,
        "passed": passed,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\n[equiv] {'PASS' if passed else 'FAIL'} -> {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    # os._exit to dodge the torch/numba CPython-finalization segfault seen in data/prepare.py
    rc = main()
    os._exit(rc)
