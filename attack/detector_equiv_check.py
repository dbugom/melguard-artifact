"""Phase-6 prep gate: forward-equivalence + gradient-to-waveform for the
differentiable detector front-end (`attack/detector_frontend.py`) vs the deployed
numpy `ASTFeatureExtractor` + frozen Phase-4 g_D.

Non-generating (loads a benign clip, computes g_D + gradients) — does NOT create
any adversarial trigger, so it does not touch the disclosure clock.

    python -m attack.detector_equiv_check [--clip PATH] [--weights out/phase4/detector_gd_best.pt]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import soundfile as sf
import torch

from attack.detector_frontend import forward_equivalence, grad_reaches_waveform
from detector.model import MelGuardDetector, detector_pins
from salmonn.loader import set_determinism

TOL = {"iv_max_abs_diff": 1e-3, "prob_abs_diff": 1e-3}


def _find_clip() -> str:
    for sub in ("clips/commonvoice18", "clips/librispeech", "clips/fleurs"):
        hits = sorted(glob.glob(os.path.join("/workspace/data", sub, "**", "*.flac"), recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError("no sample clip under /workspace/data/clips")


def _load_detector(weights: str, device: str):
    det = MelGuardDetector()
    sd = torch.load(weights, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = det.load_state_dict(sd, strict=False)
    det.eval().to(device)
    return det, {"missing": len(missing), "unexpected": len(unexpected)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=None)
    ap.add_argument("--weights", default="out/phase4/detector_gd_best.pt")
    ap.add_argument("--out", default="out/phase6/detector_frontend_equiv.json")
    args = ap.parse_args()

    seed = set_determinism()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    from transformers import ASTFeatureExtractor
    d = detector_pins()
    fe = ASTFeatureExtractor.from_pretrained(d["hf_repo"], revision=d["hf_revision"])

    clip = args.clip or _find_clip()
    wav, sr = sf.read(clip)
    if wav.ndim == 2:
        wav = wav[:, 0]
    wav = wav.astype(np.float32)
    assert sr == 16000

    print(f"[det-equiv] seed={seed} device={device} clip={clip} dur={len(wav)/sr:.2f}s")
    det, load_info = _load_detector(args.weights, device)
    print(f"[det-equiv] detector loaded (missing={load_info['missing']} unexpected={load_info['unexpected']})")

    fwd = forward_equivalence(det, fe, wav)
    grad = grad_reaches_waveform(det, wav, fe, target_len=16000)

    checks = {
        "iv_parity": fwd["iv_max_abs_diff"] <= TOL["iv_max_abs_diff"],
        "prob_parity": fwd["prob_abs_diff"] <= TOL["prob_abs_diff"],
        "grad_finite": grad["grad_is_finite"],
        "grad_nonzero": grad["grad_abs_sum"] > 0.0,
    }
    passed = all(checks.values())
    result = {
        "phase": 6, "step": "prep_differentiable_detector_frontend", "seed": seed,
        "clip": clip, "weights": args.weights, "load_info": load_info,
        "tolerances": TOL, "forward_equivalence": fwd, "grad_reaches_waveform": grad,
        "checks": checks, "passed": passed,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps({k: result[k] for k in ("forward_equivalence", "grad_reaches_waveform", "checks", "passed")}, indent=2))
    print(f"\n[det-equiv] {'PASS' if passed else 'FAIL'} -> {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    rc = main()
    os._exit(rc)
