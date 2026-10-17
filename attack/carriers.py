"""Reproducible, verifiable attack-carrier pool (spec.lock.md:35 + seal discipline).

The spec says carriers are benign CommonVoice-18 utterances (en/zh-CN/es). The
Phase-7 seal forbids touching the 10 k benign held-out (CV `test`/`validation`)
before the one-time unseal, and good practice keeps carriers out of detector
TRAINING too. Those constraints are jointly satisfiable within CommonVoice:

  * ``data/prepare.py::prep_commonvoice`` pooled only the FIRST
    ``cap_per_lang = BENIGN_TRAIN_PER_CORPUS//3 + BENIGN_TRAIN_PER_CORPUS//20``
    ( = 1533 ) streamed rows of CV `train` per language for ``benign_train``.
  * So CV `train` rows at stream index >= 1533 were pooled by NOTHING, and are a
    different split from the sealed `test`/`validation`.

We therefore take the carriers from CV `train`, skipping the first ``SKIP``
(= 1533) rows, then verify **sha256-disjointness** against every prepared clip in
``metadata/*.jsonl`` (so disjointness is guaranteed by content, not just by the
skip arithmetic). Selection is deterministic (fixed stream order + fixed skip +
PRIMARY_SEED), and a manifest is written for verifiability. This honors the
spec's "CommonVoice carrier" wording AND the seal, so no seal exception is
needed. 25 carriers/lang x 3 langs = 75 (NUMBER_LOCK carrier_prompts=25).

    python -m attack.carriers [--per-lang 25] [--force]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.config import (COMMONVOICE_REPO, DATA_ROOT, LANGS, METADATA_DIR,  # noqa: E402
                         BENIGN_TRAIN_PER_CORPUS, iter_metadata)
from data.prepare import to_mono_16k, uid  # noqa: E402  (reuse the exact helpers)

# The band prep_commonvoice pooled for benign_train (per lang). Carriers start
# strictly after it -> disjoint from the detector-train candidate pool.
SKIP_PER_LANG = BENIGN_TRAIN_PER_CORPUS // 3 + max(1, BENIGN_TRAIN_PER_CORPUS // 20)  # 1533
CARRIER_DATASET = "commonvoice18_carrier"
CARRIER_STAGE = "attack_carriers"


def _prepared_shas() -> set[str]:
    return {r["sha256"] for r in iter_metadata() if "sha256" in r}


def _cv_features():
    from datasets import Audio, Features, Value
    return Features({
        "client_id": Value("string"), "path": Value("string"),
        "sentence_id": Value("string"), "sentence": Value("string"),
        "sentence_domain": Value("string"), "up_votes": Value("string"),
        "down_votes": Value("string"), "age": Value("string"),
        "gender": Value("string"), "variant": Value("string"),
        "locale": Value("string"), "segment": Value("string"),
        "accent": Value("string"), "audio": Audio(decode=False),  # skip cheaply, decode manually
    })


def _save_flac(clip_uid: str, arr: np.ndarray) -> tuple[str, str]:
    import hashlib
    rel = f"clips/{CARRIER_DATASET}/{clip_uid}.flac"
    path = DATA_ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, arr.astype(np.float32), 16000, format="FLAC")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return rel, sha


def select_carriers(per_lang: int = 25, force: bool = False) -> list[dict]:
    from datasets import load_dataset

    stage_file = METADATA_DIR / f"{CARRIER_STAGE}.jsonl"
    if stage_file.exists() and not force:
        print(f"[carriers] {stage_file.name} exists; use --force to rebuild")
        return [json.loads(l) for l in stage_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    banned = _prepared_shas()
    print(f"[carriers] {len(banned)} prepared clip shas to avoid; skip_per_lang={SKIP_PER_LANG}")
    feats = _cv_features()
    recs: list[dict] = []
    for lang in LANGS:
        ds = load_dataset(COMMONVOICE_REPO, lang, split="train", streaming=True,
                          trust_remote_code=True, features=feats)
        ds = ds.skip(SKIP_PER_LANG)
        got = 0
        for j, row in enumerate(ds):
            if got >= per_lang:
                break
            audio = row["audio"]
            try:
                arr, sr = sf.read(io.BytesIO(audio["bytes"]))
            except Exception:  # noqa: BLE001
                continue
            arr16 = to_mono_16k(arr, sr)
            if arr16.size < int(0.5 * 16000):   # skip <0.5 s fragments
                continue
            cid = row.get("client_id", "")
            cu = uid("cvcarrier", lang, SKIP_PER_LANG + j, cid)
            rel, sha = _save_flac(cu, arr16)
            if sha in banned:                    # content-level disjointness guarantee
                print(f"  WARN carrier {cu} sha collided with a prepared clip; dropping")
                (DATA_ROOT / rel).unlink(missing_ok=True)
                continue
            banned.add(sha)
            recs.append({
                "uid": cu, "path": rel, "sha256": sha, "role": "attack_carrier",
                "dataset": CARRIER_DATASET, "language": lang,
                "seconds": round(arr16.size / 16000, 3), "stream_index": SKIP_PER_LANG + j,
                "client_id": cid,
            })
            got += 1
        print(f"  {lang}: selected {got} carriers")

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(stage_file, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    # verifiable manifest
    man = METADATA_DIR / f"{CARRIER_STAGE}.manifest.json"
    man.write_text(json.dumps({
        "n": len(recs), "per_lang": per_lang, "skip_per_lang": SKIP_PER_LANG,
        "source": f"{COMMONVOICE_REPO}:train (index >= {SKIP_PER_LANG})",
        "disjoint_from": "all metadata/*.jsonl clips (sha256-verified)",
        "by_lang": {l: sum(1 for r in recs if r["language"] == l) for l in LANGS},
    }, indent=2), encoding="utf-8")
    print(f"[carriers] wrote {len(recs)} carriers -> {stage_file.name} (+ manifest)")
    return recs


def load_carriers() -> list[dict]:
    stage_file = METADATA_DIR / f"{CARRIER_STAGE}.jsonl"
    if not stage_file.exists():
        raise FileNotFoundError("run `python -m attack.carriers` first to build the carrier pool")
    return [json.loads(l) for l in stage_file.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=25)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    recs = select_carriers(per_lang=args.per_lang, force=args.force)
    print(f"[carriers] total {len(recs)}")
    return 0


if __name__ == "__main__":
    rc = main()
    os._exit(rc)
