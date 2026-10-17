"""Phase-4 detector data: metadata records (by role) -> cached AST features.

Splits (roles set in Phase-1 metadata; see env/DATA_MANIFEST.lock.json):
  train : benign_train (LibriSpeech+CommonVoice+FLEURS, label 0) + public_train (JALMBench, label 1)
  dev   : benign_val   (LibriSpeech+CommonVoice+FLEURS, label 0) + public_dev   (JALMBench, label 1)
Benign train/dev are multi-corpus x 3 langs on disjoint splits (SPEC-CHANGE-007).
The held-out benign_auc / benign_fpr / public_held (and FLEURS benign_control)
are NEVER loaded here — no held-out clip touches training or threshold selection.

Features (ASTFeatureExtractor: 128-mel, AudioSet norm, the pinned max_length) are
precomputed once per split and cached on the volume, keyed by (split, n) so 20
epochs don't re-decode FLAC. Records are sorted by uid for deterministic order.
"""
from __future__ import annotations

import glob
import hashlib
import os
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.config import DATA_ROOT, iter_metadata  # noqa: E402
from detector.model import detector_pins  # noqa: E402

SR = 16000

# --- Phase 5/6: generated adversarial triggers as detector training positives -----
# Pre-registration 344-47/344-51: g_D's adversarial training class includes the
# "in-house clips generated under the threat model" (the train split of the cube).
# SEAL-SAFE by construction: attack-seed == train/dev/held split, so only the train
# seed (1729) and dev seed (31337) triggers are ever loaded here; the HELD seed
# (271828) is sealed for Phase 7 and is intentionally NOT in TRIGGER_SEED_BY_SPLIT.
BACKUP_TRIGGERS = Path(os.environ.get("MELGUARD_BACKUP", "/workspace/melguard-backup")) / "triggers"
TRIGGER_SEED_BY_SPLIT = {"train": 1729, "dev": 31337}
_TRIGGER_TOOLS = ("transfer_money", "ssh_exec", "open_door")
# Which generated-trigger conditions to include as detector positives. Default =
# lambda=0 only (the first hardened detector, reproducible). Set MELGUARD_TRIGGER_TAGS
# to a comma list (e.g. "phase5_lam0,phase6_lam1") for the adaptive-retrain round.
_TRIGGER_TAGS = tuple(t for t in os.environ.get("MELGUARD_TRIGGER_TAGS", "phase5_lam0").split(",") if t)

ROLES = {
    "train": ["benign_train", "public_train"],
    "dev": ["benign_val", "public_dev"],
    "control": ["benign_control"],  # FLEURS en/zh/es — cross-corpus benign control (all label 0)
}


def load_records(roles: list[str]) -> list[dict]:
    want = set(roles)
    recs = [r for r in iter_metadata() if r.get("role") in want]
    recs.sort(key=lambda r: r["uid"])  # deterministic
    return recs


def _generated_trigger_records(split: str) -> list[dict]:
    """Canonical lambda=0 generated triggers for this split's attack-seed, as
    adversarial (label-1) training positives — hardens g_D on the prepend-trigger
    attack mode (pre-reg 344-47). SEAL-SAFE: returns [] for any split not in
    TRIGGER_SEED_BY_SPLIT, so the held seed (271828) is never referenced. Keeps only
    the canonical sweep run-ids ``phase5_lam0_<tool>_<lang>_s<seed>_<uid>`` (skipping
    pilot / batch-validate / phase6 tags) and dedupes re-runs by carrier uid."""
    seed = TRIGGER_SEED_BY_SPLIT.get(split)
    if seed is None:
        return []
    sep = f"_s{seed}_"
    by_key: dict[str, dict] = {}
    for tag in _TRIGGER_TAGS:
        for fp in sorted(glob.glob(str(BACKUP_TRIGGERS / f"{tag}_*{sep}*.flac"))):
            stem = Path(fp).stem
            left, uid = stem.split(sep, 1)
            if not left.startswith(tag + "_"):
                continue
            rest = left[len(tag) + 1:]                       # <tool>_<lang>
            tool = next((t for t in _TRIGGER_TOOLS if rest.startswith(t + "_")), None)
            if tool is None:
                continue                                     # smoke / pilot / batch-validate variant
            lang = rest[len(tool) + 1:]
            if lang not in ("en", "zh-CN", "es"):
                continue                                     # skip smoke/throwaway carriers (e.g. lang="smoke")
            by_key.setdefault(f"{tag}_{uid}", {              # key by (tag, uid): same carrier at a different lambda = a distinct trigger
                "uid": f"gen_{tag}_s{seed}_{uid}",
                "path": fp,                                  # absolute; DATA_ROOT / abs == abs (pathlib)
                "class": "adversarial",
                "language": lang,
                "dataset": tag,
                "role": f"generated_{split}",
            })
    return list(by_key.values())


def _feature_extractor(ast_path: str, revision: str):
    from transformers import ASTFeatureExtractor

    return ASTFeatureExtractor.from_pretrained(ast_path, revision=revision)


def build_or_load_features(
    split: str,
    cache_dir: str | Path,
    ast_path: str | None = None,
    revision: str | None = None,
    limit: int | None = None,
    verbose: bool = True,
) -> dict:
    """Return {input_values (N,L,128) fp16, labels (N,), langs, uids} for a split."""
    d = detector_pins()
    ast_path = ast_path or d["hf_repo"]
    revision = revision or d["hf_revision"]

    recs = load_records(ROLES[split]) + _generated_trigger_records(split)
    recs.sort(key=lambda r: r["uid"])  # deterministic order incl. generated triggers
    if limit:
        # keep the split class-balanced under a smoke limit: interleave pos/neg
        pos = [r for r in recs if r["class"] == "adversarial"][: max(1, limit // 2)]
        neg = [r for r in recs if r["class"] == "benign"][: max(1, limit // 2)]
        recs = sorted(pos + neg, key=lambda r: r["uid"])

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Key on the exact record SET, not just its size. `{split}_{n}.pt` collided: a dev cache
    # built with MELGUARD_TRIGGER_TAGS=phase6_lam1 has the same 2899 count as one built with
    # phase5_lam0, so a later run silently loaded the wrong clips. Every caller was exposed;
    # only the Phase-7 driver happened to have a membership check that surfaced it.
    sig = hashlib.sha256("\n".join(r["uid"] for r in recs).encode()).hexdigest()[:12]
    cache = cache_dir / f"{split}_{len(recs)}_{sig}.pt"
    if cache.exists():
        if verbose:
            print(f"[data] {split}: loading cached features {cache.name}")
        return torch.load(cache, weights_only=False)

    fe = _feature_extractor(ast_path, revision)
    feats, labels, langs, uids, datasets = [], [], [], [], []
    for i, r in enumerate(recs):
        wav, sr = sf.read(DATA_ROOT / r["path"])
        if wav.ndim == 2:
            wav = wav[:, 0]
        if sr != SR:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR, res_type="fft")
        iv = fe(wav, sampling_rate=SR, return_tensors="pt").input_values[0]  # (L, 128)
        feats.append(iv.to(torch.float16))
        labels.append(1 if r["class"] == "adversarial" else 0)
        langs.append(r.get("language"))
        uids.append(r["uid"])
        datasets.append(r.get("dataset"))  # corpus label -> per-corpus FPR (SC-007 check)
        if verbose and (i + 1) % 1000 == 0:
            print(f"[data] {split}: featurized {i + 1}/{len(recs)}", flush=True)

    data = {
        "input_values": torch.stack(feats),
        "labels": torch.tensor(labels, dtype=torch.long),
        "langs": langs,
        "datasets": datasets,
        "uids": uids,
    }
    torch.save(data, cache)
    if verbose:
        n_pos = int(data["labels"].sum())
        print(f"[data] {split}: {len(recs)} clips ({n_pos} adv / {len(recs) - n_pos} benign) "
              f"-> cached {cache.name}")
    return data
