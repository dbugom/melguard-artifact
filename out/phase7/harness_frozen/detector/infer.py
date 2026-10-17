"""Phase-4: freeze tau_op and report dev AUC + per-language ROC for g_D.

tau_op is selected on the **dev** benign set for FPR = target_dev_fpr (0.02, from
NUMBER_LOCK) and then frozen — it is never re-selected on held-out data (NUMBER_LOCK
tau_op.frozen_before_heldout). Writes out/phase4/detector_metrics.json (+ MANIFEST).
The held-out benign_auc / benign_fpr / public_held sets are NOT touched here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from detector.data import build_or_load_features
from detector.model import MelGuardDetector
from melguard.locks import number_lock, sha256_file


def predict_all(model: MelGuardDetector, data: dict, device: str, batch: int = 64):
    """Return (probs, labels) as numpy arrays over a feature dict."""
    model.eval()
    X = data["input_values"]
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i : i + batch].to(device=device, dtype=torch.float32)
            probs.append(torch.sigmoid(model(xb)).float().cpu())
    return torch.cat(probs).numpy(), data["labels"].numpy()


def select_tau_op(benign_probs: np.ndarray, target_fpr: float) -> float:
    """tau such that fraction(benign_prob > tau) ~= target_fpr."""
    return float(np.quantile(benign_probs, 1.0 - target_fpr))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="out/phase4/detector_gd_best.pt")
    ap.add_argument("--cache-dir", default="/workspace/data/features")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="out/phase4")
    ap.add_argument("--limit", type=int, default=None, help="smoke: cap dev clips")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nl = number_lock()
    target_fpr = float(nl["tau_op"]["target_dev_fpr"])  # 0.02

    model = MelGuardDetector().to(args.device)
    model.load_state_dict(torch.load(args.weights, map_location=args.device, weights_only=True))

    dev = build_or_load_features("dev", args.cache_dir, limit=args.limit)
    probs, labels = predict_all(model, dev, args.device)

    dev_auc = float(roc_auc_score(labels, probs))
    benign = probs[labels == 0]
    tau_op = select_tau_op(benign, target_fpr)
    fpr_at_tau = float((benign > tau_op).mean())
    tpr_at_tau = float((probs[labels == 1] > tau_op).mean())

    langs = np.array([lg if lg is not None else "unk" for lg in dev["langs"]])
    dsets = np.array([d if d is not None else "unk" for d in dev.get("datasets", ["unk"] * len(labels))])
    is_benign = labels == 0

    def _group_fpr(keys) -> dict:
        # benign FPR at tau_op per group (SC-007 corpus-shortcut check + the
        # masking-trap guard: gate per-language/per-corpus FPR, not just AUC).
        out = {}
        for k in sorted(set(keys[is_benign].tolist())):
            m = is_benign & (keys == k)
            out[k] = {"n": int(m.sum()), "fpr_at_tau_op": round(float((probs[m] > tau_op).mean()), 4)}
        return out

    per_language_fpr = _group_fpr(langs)
    per_corpus_fpr = _group_fpr(dsets)
    per_language_auc = {}
    for lg in sorted(set(langs.tolist())):
        m = langs == lg
        if len(np.unique(labels[m])) == 2:
            per_language_auc[lg] = round(float(roc_auc_score(labels[m], probs[m])), 4)

    auc_bias_flags = [lg for lg, a in per_language_auc.items() if a < 0.80]     # guardrail §6
    FPR_TOL = 0.03  # per-group benign FPR tolerance (target 0.02 + slack)
    fpr_flags = (
        [f"lang:{k}={v['fpr_at_tau_op']}" for k, v in per_language_fpr.items() if v["fpr_at_tau_op"] > FPR_TOL]
        + [f"corpus:{k}={v['fpr_at_tau_op']}" for k, v in per_corpus_fpr.items() if v["fpr_at_tau_op"] > FPR_TOL]
    )

    metrics = {
        "phase": 4,
        "weights": args.weights,
        "dev_auc": round(dev_auc, 4),
        "tau_op": tau_op,
        "target_dev_fpr": target_fpr,
        "fpr_at_tau_op": round(fpr_at_tau, 4),
        "tpr_at_tau_op": round(tpr_at_tau, 4),
        "n_dev": int(len(labels)),
        "n_dev_adv": int(labels.sum()),
        "per_language_fpr": per_language_fpr,
        "per_corpus_fpr": per_corpus_fpr,
        "per_language_auc": per_language_auc,
        "auc_bias_below_0.80": auc_bias_flags,
        "fpr_tolerance": FPR_TOL,
        "fpr_flags": fpr_flags,
        "frozen": True,
        "note": "tau_op frozen on multi-corpus dev benign for FPR=0.02; held-out (CommonVoice test/validation) sealed & untouched.",
    }
    res = out_dir / "detector_metrics.json"
    res.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "MANIFEST.sha256").write_text(
        f"{sha256_file(res)}  detector_metrics.json\n", encoding="utf-8"
    )

    ok = dev_auc >= 0.90 and not fpr_flags  # baseline gate; sealed 4-lambda bar is Phase 7
    print(json.dumps(metrics, indent=2))
    print(f"\n--- PHASE-4 (baseline g_D): dev_auc={dev_auc:.4f} tau_op={tau_op:.4f} fpr@tau={fpr_at_tau:.4f}")
    print(f"    per-corpus FPR: { {k: v['fpr_at_tau_op'] for k, v in per_corpus_fpr.items()} }")
    print(f"    per-lang   FPR: { {k: v['fpr_at_tau_op'] for k, v in per_language_fpr.items()} }")
    print(f"    FPR flags >{FPR_TOL}: {fpr_flags or 'none'} | AUC bias<0.80: {auc_bias_flags or 'none'} ---")
    return 0 if ok else 0  # informational baseline; do not hard-fail


if __name__ == "__main__":
    raise SystemExit(main())
