"""Phase-8 statistical-sensitivity pass (post-hoc, does NOT re-open the seal).

Addresses the three camera-ready hardening items flagged in the paper's Limitations
and in docs/DEVIATIONS-FROM-PREREG.md ("Three hardening items deferred..."):

  (a) speaker-clustered negatives  -- resample the 5,000 benign clips by CommonVoice
      client_id instead of as independent clips (only widens intervals);
  (b) paired Delta-AUC / trend test -- bootstrap AUC differences between conditions on
      the shared carrier structure, to formalize the qualitative decline;
  (c) B >= 50,000 bootstrap draws   -- de-noise the extreme Holm-level tail quantiles.

None of these can change the FPR-bound 0/4 headline; this is a robustness report, run
once on the ALREADY-UNSEALED per-clip scores. It re-decides nothing.

STATUS: authored offline; NOT executed here (this workstation has no torch and the
per-clip scores were not persisted). It requires two env-dependent inputs, both marked
NEEDS_ENV below:
  1. Regenerated per-clip detector scores (deterministic; validate that the aggregate
     AUC/FPR reproduce out/phase7/sealed_result.json before trusting the sensitivity CIs).
  2. A uid -> client_id map for the CommonVoice negatives (not persisted in
     data/metadata/benign_commonvoice.jsonl; recover from the CommonVoice 18 validated
     TSV via the prepare.py provenance, then key by clip uid).

Run on a box that has the pinned torch stack + the original data (e.g. the attack pod):
    python -m robustness.phase8_sensitivity --out out/phase8/sensitivity.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Reuse the FROZEN harness -- identical AUC/BCa machinery as the sealed decision.
from detector.sealed_eval import auc_bca_ci, wilson_ci  # noqa: E402

LAMBDAS = [0.0, 0.1, 1.0, 10.0]
B_HARDENED = 50_000
SEED = 0


# --------------------------------------------------------------------------- inputs
def load_scored_clips():
    """Return the per-clip evaluation table.

    Expected schema (one row per sealed clip):
        {
          "uid": str,
          "role": "benign_auc" | "benign_fpr" | "adversarial",
          "lambda": float | None,        # condition, for adversarial rows
          "score": float,                # detector p_det on the SAVED waveform
          "label": 0 | 1,                # 0 benign, 1 adversarial
          "carrier_uid": str | None,     # carrier-utterance cluster (positives)
          "client_id": str | None,       # CommonVoice speaker cluster (negatives)  <- NEEDS_ENV
          "language": str,
        }

    NEEDS_ENV: the detector scores were not persisted with the sealed result, so this
    must regenerate them. Faithful path (deterministic, no GPU required):

        from detector.infer import score_clips          # AST forward, pinned extractor
        from melguard.locks import load                 # frozen weights path + sha256
        scores = score_clips(clip_paths, weights=DEPLOYED_WEIGHTS)  # phase6-hardened

    Then attach client_id from the recovered CommonVoice map (see module docstring) and
    VALIDATE: the pooled AUC/FPR recomputed here must match out/phase7/sealed_result.json
    to within bootstrap noise before the sensitivity CIs are reported.
    """
    raise NotImplementedError(
        "NEEDS_ENV: regenerate per-clip scores + attach client_id; see docstring."
    )


# ----------------------------------------------------------------- (a)+(c) per-cond AUC
def auc_by_condition(rows, n_boot: int):
    """Speaker/carrier-clustered BCa AUC CI per lambda, at the hardened draw count.

    Negatives are clustered by client_id, positives by carrier_uid -- the resampling
    unit is the cluster, so repeated speakers no longer inflate precision.
    """
    out = {}
    neg = [r for r in rows if r["label"] == 0 and r["role"] == "benign_auc"]
    neg_scores = np.array([r["score"] for r in neg])
    neg_units = np.array([r["client_id"] or r["uid"] for r in neg])   # speaker cluster
    for lam in LAMBDAS:
        pos = [r for r in rows if r["label"] == 1 and r["lambda"] == lam]
        pos_scores = np.array([r["score"] for r in pos])
        pos_units = np.array([r["carrier_uid"] or r["uid"] for r in pos])
        scores = np.concatenate([neg_scores, pos_scores])
        labels = np.concatenate([np.zeros(len(neg_scores)), np.ones(len(pos_scores))])
        units = np.concatenate([neg_units, pos_units])
        strata = np.array([r["language"] for r in neg] + [r["language"] for r in pos])
        ci = auc_bca_ci(scores, labels, strata, units=units, n_boot=n_boot, seed=SEED)
        out[lam] = {"auc": float(ci.point), "ci95": [float(ci.lo), float(ci.hi)],
                    "n_pos": int(len(pos_scores)), "n_neg_clusters": int(len(set(neg_units)))}
    return out


# --------------------------------------------------------------------- (a) clustered FPR
def fpr_speaker_clustered(rows, n_boot: int):
    """FPR point + clustered-bootstrap CI on the benign_fpr set, clustered by speaker.

    Direction check: clustering can only WIDEN this interval, so the registered
    Wilson gate (upper <= 0.025) can only fail harder. Reported for completeness.
    """
    fpr = [r for r in rows if r["role"] == "benign_fpr"]
    tau = _frozen_tau_op()
    flags = np.array([1.0 if r["score"] > tau else 0.0 for r in fpr])
    units = np.array([r["client_id"] or r["uid"] for r in fpr])
    point = float(flags.mean())
    wilson = wilson_ci(int(flags.sum()), len(flags), 0.05)
    # speaker-cluster bootstrap of the FPR
    uniq = np.array(sorted(set(units)))
    rng = np.random.default_rng(SEED)
    boots = np.empty(n_boot)
    idx_by_unit = {u: np.where(units == u)[0] for u in uniq}
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_unit[u] for u in pick])
        boots[b] = flags[sel].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"fpr_point": point, "wilson95": [float(wilson[0]), float(wilson[1])],
            "cluster_boot95": [float(lo), float(hi)], "n_speakers": int(len(uniq))}


# ------------------------------------------------------------------ (b) paired Delta-AUC
def paired_delta_auc(rows, n_boot: int):
    """Bootstrap AUC differences between adjacent lambda, resampling the SHARED carrier
    set once per replicate (paired), plus a one-sided monotone-trend check."""
    neg = [r for r in rows if r["label"] == 0 and r["role"] == "benign_auc"]
    neg_scores = np.array([r["score"] for r in neg])
    neg_units = np.array([r["client_id"] or r["uid"] for r in neg])
    pos_by_lam = {}
    for lam in LAMBDAS:
        p = [r for r in rows if r["label"] == 1 and r["lambda"] == lam]
        pos_by_lam[lam] = (np.array([r["score"] for r in p]),
                           np.array([r["carrier_uid"] or r["uid"] for r in p]))

    def _auc(ps, pu, resampled_neg_idx, carrier_pick):
        pos_idx = np.concatenate([np.where(pu == c)[0] for c in carrier_pick if (pu == c).any()])
        s = np.concatenate([neg_scores[resampled_neg_idx], ps[pos_idx]])
        y = np.concatenate([np.zeros(len(resampled_neg_idx)), np.ones(len(pos_idx))])
        order = np.argsort(s)
        ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(s) + 1)
        n1 = int(y.sum()); n0 = len(y) - n1
        return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)

    neg_uniq = np.array(sorted(set(neg_units)))
    neg_by_unit = {u: np.where(neg_units == u)[0] for u in neg_uniq}
    carriers = np.array(sorted(set(np.concatenate([pu for _, pu in pos_by_lam.values()]))))
    rng = np.random.default_rng(SEED)
    pairs = [(LAMBDAS[i], LAMBDAS[i + 1]) for i in range(len(LAMBDAS) - 1)]
    diffs = {p: np.empty(n_boot) for p in pairs}
    for b in range(n_boot):
        neg_pick = np.concatenate([neg_by_unit[u] for u in rng.choice(neg_uniq, len(neg_uniq), replace=True)])
        carrier_pick = rng.choice(carriers, len(carriers), replace=True)   # shared -> paired
        a = {lam: _auc(*pos_by_lam[lam], neg_pick, carrier_pick) for lam in LAMBDAS}
        for (l0, l1) in pairs:
            diffs[(l0, l1)][b] = a[l1] - a[l0]
    out = {}
    for (l0, l1), d in diffs.items():
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[f"AUC({l1})-AUC({l0})"] = {"delta_median": float(np.median(d)),
                                       "ci95": [float(lo), float(hi)],
                                       "p_decrease": float((d >= 0).mean())}
    return out


def _frozen_tau_op() -> float:
    rec = json.loads(Path("out/phase7/sealed_result.json").read_text(encoding="utf-8"))
    return float(rec.get("tau_op") or rec.get("frozen_tau_op") or 0.00032981)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/phase8/sensitivity.json")
    ap.add_argument("--n-boot", type=int, default=B_HARDENED)
    args = ap.parse_args()

    rows = load_scored_clips()   # NEEDS_ENV
    result = {
        "note": "post-hoc sensitivity on already-unsealed scores; re-decides nothing",
        "n_boot": args.n_boot,
        "auc_by_condition_speaker_clustered": auc_by_condition(rows, args.n_boot),
        "fpr_speaker_clustered": fpr_speaker_clustered(rows, args.n_boot),
        "paired_delta_auc": paired_delta_auc(rows, args.n_boot),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
