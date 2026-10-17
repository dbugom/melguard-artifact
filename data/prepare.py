"""Phase-1 data preparation pipeline (resumable, deterministic).

Produces 16 kHz mono FLAC clips + per-clip metadata JSONL for:
  benign_train (~12k)         <- LibriSpeech train.100 (en) + CommonVoice-18 `train`
                                 + FLEURS `train` (en/zh/es) — multi-corpus (SPEC-CHANGE-007)
  benign_val   (~2.7k)        <- LibriSpeech dev-clean + CommonVoice-18 `other`
                                 + FLEURS `validation` (en/zh/es) — for tau_op (SPEC-CHANGE-007)
  benign_auc  / benign_fpr    <- CommonVoice 18 (fsicoli mirror) `test` / `validation`,
                                 en/zh-CN/es, 5,000 + 5,000 disjoint (sealed held-out, UNCHANGED)
  benign_control (~900)       <- FLEURS `test` (en/zh/es) — non-sealed early-warning (SPEC-CHANGE-007)
  public_train/dev/held       <- JALMBench audio-originated {AdvWave,SSJ,BoN,AMSE}, split by attack-id hash

All corpora share ONE decode->resample path (native-sr decode + fixed-param
resampler) so the resample signature no longer tracks the label (SPEC-CHANGE-008).

Guardrail: this is BENIGN + PUBLIC adversarial only. It downloads existing public
clips; it never generates or exports a working trigger waveform (that is Phases 5-6,
hashes-only). Nothing here contradicts spec §10.

CLI (each stage is idempotent; re-runs skip existing metadata unless --force):
  python data/prepare.py librispeech [--limit N]
  python data/prepare.py commonvoice [--limit N] [--pool-per-lang M]
  python data/prepare.py fleurs      [--limit N]
  python data/prepare.py jalmbench   [--limit N]
  python data/prepare.py all
  python data/manifests.py build      # then build the SHA-256 manifest
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from melguard.locks import sha256_file  # noqa: E402

from data.config import (  # noqa: E402
    BENIGN_AUC_N, BENIGN_CONTROL_PER_LANG, BENIGN_FPR_N, BENIGN_TRAIN_PER_CORPUS,
    BENIGN_VAL_PER_CORPUS, CLIPS_DIR, COMMONVOICE_REPO, DATA_ROOT, FLEURS_LANG,
    FLEURS_REPO, JALMBENCH_AUDIO_CONFIGS, JALMBENCH_REPO, LANGS, LIBRISPEECH_REPO,
    METADATA_DIR, PRIMARY_SEED, PUBLIC_ADV_TOTAL, PUBLIC_SPLIT, RESAMPLE_RES_TYPE,
    SR, snr_bucket,
)

# --------------------------------------------------------------------------- #
# audio helpers
# --------------------------------------------------------------------------- #

def estimate_snr_db(x: np.ndarray) -> float:
    """Cheap energy-percentile SNR proxy for stratification (not a physical SNR)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    frame = 400  # 25 ms @ 16 kHz
    n = (len(x) // frame) * frame
    if n < frame:
        e = np.array([np.mean(x ** 2) + 1e-12])
    else:
        e = np.mean(x[:n].reshape(-1, frame) ** 2, axis=1) + 1e-12
    noise = max(float(np.percentile(e, 10)), 1e-12)
    sig = max(float(np.percentile(e, 90)), 1e-12)
    return float(10.0 * np.log10(sig / noise))


def to_mono_16k(arr, sr: int) -> np.ndarray:
    """Downmix to mono and resample to 16 kHz through ONE shared path.

    SPEC-CHANGE-008: every corpus decodes at its native sr (callers use
    Audio(decode=True) with NO per-corpus sampling_rate cast) and is resampled by
    the same librosa resampler with fixed params (RESAMPLE_RES_TYPE) for EVERY
    clip. There is deliberately no `sr == SR` fast-path: the decode/resample path
    is identical for all corpora, so "was-librosa-resampled" can no longer
    correlate with the adversarial label. (librosa short-circuits the 16k->16k
    identity internally; the point is that the code path and params are uniform.)
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1 if arr.shape[0] > arr.shape[-1] else 0)
    import librosa
    arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=SR,
                           res_type=RESAMPLE_RES_TYPE)
    return np.clip(arr.astype(np.float32), -1.0, 1.0)


def uid(*parts) -> str:
    return hashlib.sha256("::".join(str(p) for p in parts).encode()).hexdigest()[:20]


def save_flac(dataset: str, clip_uid: str, arr: np.ndarray) -> tuple[str, str]:
    rel = f"clips/{dataset}/{clip_uid}.flac"
    p = DATA_ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        sf.write(str(p), np.clip(arr, -1.0, 1.0), SR, format="FLAC", subtype="PCM_16")
    return rel, sha256_file(p)


def stage_path(stage: str) -> Path:
    return METADATA_DIR / f"{stage}.jsonl"


def write_stage(stage: str, records: list[dict]) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with stage_path(stage).open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"  wrote {len(records)} records -> {stage_path(stage).name}")


def stage_done(stage: str, force: bool) -> bool:
    if force:
        return False
    p = stage_path(stage)
    return p.exists() and p.stat().st_size > 0


def make_record(dataset, role, clip_uid, rel, sha, arr, language,
                gender=None, attack_family=None, instruction_id=None) -> dict:
    dur = round(len(arr) / SR, 3)
    return {
        "uid": clip_uid, "path": rel, "sha256": sha,
        "class": "adversarial" if attack_family else "benign",
        "role": role, "dataset": dataset, "language": language,
        "gender": gender, "snr_bucket": snr_bucket(estimate_snr_db(arr)),
        "duration_s": dur, "sample_rate": SR,
        "attack_family": attack_family, "instruction_id": instruction_id,
    }


# --------------------------------------------------------------------------- #
# benign stages
# --------------------------------------------------------------------------- #

def prep_librispeech(limit: int | None, force: bool) -> None:
    # SC-007: LibriSpeech stays en-only; it is one of the three benign corpora.
    #   train.100 -> benign_train (cap BENIGN_TRAIN_PER_CORPUS),
    #   dev-clean (HF "validation") -> benign_val (cap BENIGN_VAL_PER_CORPUS).
    # SC-008: decode at native sr (Audio(decode=True), NO sampling_rate cast);
    #   to_mono_16k applies the single shared resampler.
    from datasets import Audio, load_dataset
    stage = "benign_librispeech"
    if stage_done(stage, force):
        print(f"[skip] {stage} already present"); return
    recs = []
    for hf_split, role, full_cap in [
        ("train.100", "benign_train", BENIGN_TRAIN_PER_CORPUS),
        ("validation", "benign_val", BENIGN_VAL_PER_CORPUS),
    ]:
        cap = limit or full_cap
        ds = load_dataset(LIBRISPEECH_REPO, "clean", split=hf_split, streaming=True)
        ds = ds.cast_column("audio", Audio(decode=True))  # native sr (SC-008)
        for i, row in enumerate(ds):
            if i >= cap:
                break
            arr = to_mono_16k(row["audio"]["array"], row["audio"]["sampling_rate"])
            cu = uid("ls", hf_split, row.get("id", i))
            rel, sha = save_flac("librispeech", cu, arr)
            recs.append(make_record("librispeech", role, cu, rel, sha, arr, "en"))
        print(f"  {role}: {sum(1 for r in recs if r['role'] == role)} clips (librispeech/{hf_split})")
    write_stage(stage, recs)


def prep_commonvoice(limit: int | None, pool_per_lang: int, force: bool) -> None:
    """Benign from CommonVoice 18 (fsicoli mirror, SPEC-CHANGE-003 + 007).

    SPEC-CHANGE-007 CV split map (all disjoint by CV construction, so the sealed
    held-out is untouched):
      train      -> benign_train  (multi-corpus benign train, ~1333/lang)
      other      -> benign_val     (dev, for tau_op, ~300/lang)
      test       -> benign_auc      (sealed held-out AUC set, 5000 — UNCHANGED)
      validation -> benign_fpr       (sealed held-out FPR set, 5000 — UNCHANGED)
    Each set is language round-robin balanced across en/zh-CN/es.

    SPEC-CHANGE-008: audio decodes at native sr (Audio(decode=True), no
    sampling_rate cast) and to_mono_16k applies the single shared resampler.

    Held-out preservation: benign_auc/benign_fpr use `rng_heldout`, seeded and
    consumed (test then validation) EXACTLY as the pre-SC-007 pipeline did — same
    PRIMARY_SEED, same caps, same round-robin, same uid formula -> byte-for-byte
    the same rows. The new SC-007 train/val collects use a SEPARATE rng instance
    (`rng_multi`), so they cannot perturb the held-out row selection whatever the
    call order.
    """
    from collections import Counter
    from datasets import Audio, Features, Value, load_dataset
    stage = "benign_commonvoice"
    if stage_done(stage, force):
        print(f"[skip] {stage} already present"); return
    # The fsicoli CV18 loader script declares a stale feature set; pass the ACTUAL
    # column schema so the cast is a no-op (audio decoded at native sr, resampled
    # to 16k below). CV18 tsv added sentence_id / sentence_domain.
    _cv_feats = Features({
        "client_id": Value("string"), "path": Value("string"),
        "sentence_id": Value("string"), "sentence": Value("string"),
        "sentence_domain": Value("string"), "up_votes": Value("string"),
        "down_votes": Value("string"), "age": Value("string"),
        "gender": Value("string"), "variant": Value("string"),
        "locale": Value("string"), "segment": Value("string"),
        "accent": Value("string"), "audio": Audio(decode=True),
    })

    def collect(split_name: str, role: str, target_total: int,
                rng: random.Random) -> list[dict]:
        cap_per_lang = limit or (target_total // len(LANGS) + max(1, target_total // 20))
        pools: dict[str, list] = {lang: [] for lang in LANGS}
        for lang in LANGS:
            try:
                # CommonVoice mirrors ship the standard CV loader script
                # (unpacks the audio tarball) -> trust_remote_code required.
                ds = load_dataset(COMMONVOICE_REPO, lang, split=split_name,
                                  streaming=True, trust_remote_code=True,
                                  features=_cv_feats)
            except Exception as e:  # noqa: BLE001
                print(f"  WARN commonvoice {lang}/{split_name}: {type(e).__name__}: {str(e)[:80]}")
                continue
            for row in ds:
                if len(pools[lang]) >= cap_per_lang:
                    break
                arr = to_mono_16k(row["audio"]["array"], row["audio"]["sampling_rate"])
                pools[lang].append((row.get("client_id", ""), row.get("gender"), arr))
            print(f"  {role}: pool[{lang}] = {len(pools[lang])}")
        for lang in LANGS:
            rng.shuffle(pools[lang])
        need = target_total if not limit else min(target_total, cap_per_lang * len(LANGS))
        combined, idx = [], {lang: 0 for lang in LANGS}
        while len(combined) < need and any(idx[l] < len(pools[l]) for l in LANGS):
            for l in LANGS:
                if idx[l] < len(pools[l]):
                    combined.append((l, *pools[l][idx[l]])); idx[l] += 1
                    if len(combined) >= need:
                        break
        if len(combined) < need:
            print(f"  WARN {role}: only {len(combined)} clips (< {need}); split may be small")
        recs = []
        for i, (lang, cid, gender, arr) in enumerate(combined):
            cu = uid("cv", role, lang, i, cid)
            rel, sha = save_flac("commonvoice18", cu, arr)
            recs.append(make_record("commonvoice18", role, cu, rel, sha, arr, lang, gender=gender))
        return recs

    # Sealed held-out (benign_auc <- test, benign_fpr <- validation): rng seeded
    # and consumed in the SAME order as the original single-rng pipeline -> the
    # selected rows are identical (only the SC-008 decode path re-hashes them).
    rng_heldout = random.Random(PRIMARY_SEED)
    recs = (collect("test", "benign_auc", BENIGN_AUC_N, rng_heldout)
            + collect("validation", "benign_fpr", BENIGN_FPR_N, rng_heldout))
    # SC-007 new multi-corpus benign train/val on an INDEPENDENT rng instance so
    # the held-out selection above is provably untouched by these collects.
    rng_multi = random.Random(PRIMARY_SEED)
    recs += (collect("train", "benign_train", BENIGN_TRAIN_PER_CORPUS, rng_multi)
             + collect("other", "benign_val", BENIGN_VAL_PER_CORPUS, rng_multi))
    write_stage(stage, recs)
    print(f"  cv roles: {dict(Counter(r['role'] for r in recs))}")


def prep_fleurs(limit: int | None, force: bool) -> None:
    # SC-007: FLEURS spans 3 langs x 3 DISJOINT splits ->
    #   train      -> benign_train   (~1333/lang),
    #   validation -> benign_val      (~300/lang),
    #   test       -> benign_control   (300/lang; non-sealed early-warning, replaces
    #                                   the old FLEURS-validation control now in val).
    # SC-008: decode at native sr (Audio(decode=True), NO sampling_rate cast);
    #   to_mono_16k applies the single shared resampler.
    from datasets import Audio, load_dataset
    stage = "benign_fleurs"
    if stage_done(stage, force):
        print(f"[skip] {stage} already present"); return
    recs = []
    split_roles = [
        ("train", "benign_train", limit or (BENIGN_TRAIN_PER_CORPUS // len(LANGS))),
        ("validation", "benign_val", limit or (BENIGN_VAL_PER_CORPUS // len(LANGS))),
        ("test", "benign_control", limit or BENIGN_CONTROL_PER_LANG),
    ]
    for hf_split, role, per_lang in split_roles:
        for lang, cfg in FLEURS_LANG.items():
            try:
                ds = load_dataset(FLEURS_REPO, cfg, split=hf_split, streaming=True)
                ds = ds.cast_column("audio", Audio(decode=True))  # native sr (SC-008)
            except Exception as e:  # noqa: BLE001
                print(f"  WARN fleurs {cfg}/{hf_split}: {type(e).__name__}: {str(e)[:100]}")
                continue
            n = 0
            for i, row in enumerate(ds):
                if i >= per_lang:
                    break
                arr = to_mono_16k(row["audio"]["array"], row["audio"]["sampling_rate"])
                cu = uid("fl", cfg, hf_split, i)  # split in uid: disjoint across roles
                rel, sha = save_flac("fleurs", cu, arr)
                recs.append(make_record("fleurs", role, cu, rel, sha, arr, lang))
                n += 1
            print(f"  {role}: fleurs/{cfg}/{hf_split} = {n}")
    write_stage(stage, recs)


# --------------------------------------------------------------------------- #
# public adversarial stage (JALMBench audio-originated) — SPEC-CHANGE-004
# --------------------------------------------------------------------------- #

def _jalmbench_rows(cfg: str, need: int):
    """Yield (audio_array, sr, instruction_key, clip_id) rows for a config.
    Robust to the streaming quirk (e.g. AdvWave) via parquet-direct fallback."""
    from datasets import load_dataset
    # 1) try streaming
    try:
        dd = load_dataset(JALMBENCH_REPO, cfg, streaming=True)
        split = list(dd.keys())[0]
        got = 0
        for row in dd[split]:
            a = row["audio"]
            instr = str(row.get("original_text") or row.get("prompt") or row.get("id"))
            yield np.asarray(a["array"], dtype=np.float32), int(a["sampling_rate"]), instr, row.get("id", got)
            got += 1
            if got >= need:
                return
        if got > 0:
            return
    except Exception as e:  # noqa: BLE001
        print(f"    streaming {cfg} failed ({type(e).__name__}); parquet fallback")
    # 2) parquet-direct fallback
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = [f for f in api.list_repo_files(JALMBENCH_REPO, repo_type="dataset")
             if f"/{cfg}/" in f and f.endswith(".parquet")]
    got = 0
    for pf in sorted(files):
        local = hf_hub_download(JALMBENCH_REPO, pf, repo_type="dataset")
        tbl = pq.read_table(local).to_pandas()
        for _, r in tbl.iterrows():
            a = r["audio"]
            if isinstance(a, dict) and a.get("bytes"):
                arr, sr = sf.read(io.BytesIO(a["bytes"]))
            elif isinstance(a, dict) and "array" in a:
                arr, sr = a["array"], a["sampling_rate"]
            else:
                continue
            instr = str(r.get("original_text") or r.get("prompt") or r.get("id"))
            yield np.asarray(arr, dtype=np.float32), int(sr), instr, r.get("id", got)
            got += 1
            if got >= need:
                return


def prep_jalmbench(limit: int | None, force: bool) -> None:
    from collections import Counter, defaultdict
    stage = "adversarial_jalmbench"
    if stage_done(stage, force):
        print(f"[skip] {stage} already present"); return
    total = limit * len(JALMBENCH_AUDIO_CONFIGS) if limit else PUBLIC_ADV_TOTAL
    per_cfg = max(1, total // len(JALMBENCH_AUDIO_CONFIGS))

    # 1. collect every clip with its (family::instruction) group key.
    #    SC-008: positives decode at native sr and flow through the SAME to_mono_16k
    #    shared resampler as every benign corpus (no special-casing) — unchanged
    #    logic, uniform decode path.
    collected = []
    for cfg in JALMBENCH_AUDIO_CONFIGS:
        n = 0
        for arr, sr, instr, cid in _jalmbench_rows(cfg, per_cfg):
            arr16 = to_mono_16k(arr, sr)
            collected.append({"arr": arr16, "key": str(cid),
                              "cfg": cfg, "cid": cid, "n": n})
            n += 1
        print(f"  {cfg}: {n} clips")

    # 2. group by harmful-instruction id (shared across families) so every
    #    rendering of an instruction stays in one split (leakage control), then
    #    assign groups largest-first to the role with the biggest remaining
    #    deficit (LPT balancing) -> holds ~70/15/15 by CLIP count even when
    #    group sizes are uneven (naive cumulative greedy skips a split when a
    #    large group straddles a boundary).
    groups: dict[str, list] = defaultdict(list)
    for item in collected:
        groups[item["key"]].append(item)
    total_n = len(collected)
    targets = {"public_train": PUBLIC_SPLIT[0] * total_n,
               "public_dev": PUBLIC_SPLIT[1] * total_n,
               "public_held": PUBLIC_SPLIT[2] * total_n}
    filled = {r: 0 for r in targets}
    role_of = {}
    for k in sorted(groups, key=lambda k: (-len(groups[k]), k)):
        r = max(targets, key=lambda r: targets[r] - filled[r])
        role_of[k] = r
        filled[r] += len(groups[k])
    print(f"  groups={len(groups)} max_group_size={max(len(v) for v in groups.values())}")

    # 3. save with assigned roles
    recs = []
    for item in collected:
        cu = uid("jb", item["cfg"], item["cid"], item["n"])
        rel, sha = save_flac("jalmbench", cu, item["arr"])
        recs.append(make_record("jalmbench", role_of[item["key"]], cu, rel, sha,
                                item["arr"], "en", attack_family=item["cfg"],
                                instruction_id=str(item["cid"])))
    write_stage(stage, recs)
    print(f"  split: {dict(Counter(r['role'] for r in recs))}")


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["librispeech", "commonvoice", "fleurs", "jalmbench", "all"])
    ap.add_argument("--limit", type=int, default=None, help="per-source cap for smoke tests")
    ap.add_argument("--pool-per-lang", type=int, default=4000, help="CommonVoice candidates/lang")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print(f"DATA_ROOT = {DATA_ROOT}")
    if args.stage in ("librispeech", "all"):
        prep_librispeech(args.limit, args.force)
    if args.stage in ("commonvoice", "all"):
        prep_commonvoice(args.limit, args.pool_per_lang, args.force)
    if args.stage in ("fleurs", "all"):
        prep_fleurs(args.limit, args.force)
    if args.stage in ("jalmbench", "all"):
        prep_jalmbench(args.limit, args.force)
    print("done. next: python data/manifests.py build")


if __name__ == "__main__":
    main()
    # torch/numba can segfault at CPython finalization on some pods; all data is
    # flushed to disk by now, so exit hard to avoid a spurious non-zero exit code.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
