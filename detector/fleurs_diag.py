"""Phase-4 diagnostic: FLEURS cross-corpus / cross-lingual benign FPR at frozen tau_op.

FLEURS (benign_control, en/zh/es) is a THIRD corpus, disjoint from the LibriSpeech
train/dev and NOT sealed. Scoring the frozen g_D on it at the frozen tau_op gives an
early read on whether the detector generalises across corpora/languages — i.e. whether
the perfect LibriSpeech-vs-JALMBench dev AUC is a corpus shortcut that will inflate FPR
on the (sealed) CommonVoice held-out. This is READ-ONLY: tau_op is loaded, never retuned
(retuning on the control would leak). The sealed CommonVoice sets are not touched.

FLEURS is all benign (label 0), so there is no AUC here — only FPR = P(prob > tau_op).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from detector.data import build_or_load_features
from detector.infer import predict_all
from detector.model import MelGuardDetector
from melguard.locks import sha256_file


def _stats(p: np.ndarray) -> dict:
    return {
        "mean_prob": float(p.mean()),
        "p50": float(np.quantile(p, 0.50)),
        "p98": float(np.quantile(p, 0.98)),
        "max": float(p.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="out/phase4/detector_gd_best.pt")
    ap.add_argument("--metrics", default="out/phase4/detector_metrics.json")
    ap.add_argument("--cache-dir", default="/workspace/data/features")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="out/phase4")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.out if not Path(args.out).is_absolute() else Path(args.out)

    tau_op = float(json.loads(Path(args.metrics).read_text())["tau_op"])  # frozen on dev

    model = MelGuardDetector().to(args.device)
    model.load_state_dict(torch.load(args.weights, map_location=args.device, weights_only=True))

    control = build_or_load_features("control", args.cache_dir)
    probs, labels = predict_all(model, control, args.device)
    assert int(labels.sum()) == 0, "FLEURS control must be all-benign (label 0)"

    langs = np.array([lg if lg is not None else "unk" for lg in control["langs"]])
    per_language = {}
    for lg in sorted(set(langs)):
        m = langs == lg
        per_language[lg] = {
            "n": int(m.sum()),
            "fpr_at_tau_op": round(float((probs[m] > tau_op).mean()), 4),
            **{k: round(v, 6) for k, v in _stats(probs[m]).items()},
        }

    result = {
        "phase": 4,
        "diagnostic": "fleurs_control",
        "tau_op": tau_op,
        "target_fpr": 0.02,
        "n": int(len(probs)),
        "overall_fpr_at_tau_op": round(float((probs > tau_op).mean()), 4),
        "prob_stats": {k: round(v, 6) for k, v in _stats(probs).items()},
        "per_language": per_language,
        "note": "FLEURS = benign control (all label 0); FPR only. tau_op frozen on dev, "
                "NOT retuned here. CommonVoice held-out remains sealed.",
    }
    res = out_dir / "fleurs_control_diag.json"
    res.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    ov = result["overall_fpr_at_tau_op"]
    print(f"\n--- FLEURS control: overall FPR@tau_op={ov:.4f} (target 0.02) | "
          f"per-lang FPR={ {k: v['fpr_at_tau_op'] for k, v in per_language.items()} } ---")
    print(f"    (sha256 {sha256_file(res)[:16]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
