"""Phase-1 data config. All counts derive from env/NUMBER_LOCK.lock.json.

Layout on disk (DATA_ROOT defaults to /workspace/data, override via MELGUARD_DATA):
  DATA_ROOT/
    clips/<dataset>/<uid>.flac      # 16 kHz mono FLAC (gitignored; lives on volume+R2)
    metadata/<stage>.jsonl          # one JSON record per clip (see prepare.py)
    MANIFEST.sha256                 # coreutils-style hash list (built by manifests.py)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from melguard.locks import number_lock  # noqa: E402

NL = number_lock()

# --- audio ---
SR = NL["trigger_budget"]["sample_rate_hz"]          # 16000
CHANNELS = 1

# --- languages ---
LANGS = ["en", "zh-CN", "es"]                        # CommonVoice locale codes
FLEURS_LANG = {"en": "en_us", "zh-CN": "cmn_hans_cn", "es": "es_419"}

# --- locked counts (NUMBER_LOCK; the sealed held-out — UNCHANGED by SC-007/008) ---
BENIGN_AUC_N = NL["held_out"]["benign_auc"]          # 5000
BENIGN_FPR_N = NL["held_out"]["benign_fpr_eval"]     # 5000
PUBLIC_ADV_TOTAL = NL["public_corpus"]["post_dedup_total_approx"]  # ~825
PUBLIC_SPLIT = (0.70, 0.15, 0.15)                    # train / dev / held by attack-id hash
SEEDS = NL["attack_seeds"]                           # [1729, 31337, 271828]
PRIMARY_SEED = SEEDS[0]

# --- SC-007 multi-corpus benign targets (NOT in NUMBER_LOCK; see SPEC-CHANGE-007).
# benign_train/benign_val now span all 3 corpora x 3 langs on DISJOINT splits so the
# sealed CommonVoice held-out (benign_auc/benign_fpr) is never touched. Per-corpus
# counts (LibriSpeech is en-only; CV/FLEURS round-robin balance en/zh-CN/es):
#   benign_train ~= 3 x BENIGN_TRAIN_PER_CORPUS = ~12000
#   benign_val   ~= 3 x BENIGN_VAL_PER_CORPUS   = ~2700 (used for tau_op)
#   benign_control = 3 x BENIGN_CONTROL_PER_LANG = ~900 (FLEURS `test`, non-sealed)
BENIGN_TRAIN_PER_CORPUS = 4000   # LibriSpeech `train.100` / CV `train` / FLEURS `train`
BENIGN_VAL_PER_CORPUS = 900      # LibriSpeech `dev-clean` / CV `other` / FLEURS `validation`
BENIGN_CONTROL_PER_LANG = 300    # FLEURS `test`, 300/lang -> ~900 early-warning control

# --- adversarial source (SPEC-CHANGE-004) ---
JALMBENCH_REPO = "AnonymousUser000/JALMBench"
JALMBENCH_AUDIO_CONFIGS = ["AdvWave", "SSJ", "BoN", "AMSE"]   # audio-originated attack families

# --- benign source (SPEC-CHANGE-003) ---
COMMONVOICE_REPO = "fsicoli/common_voice_18_0"
LIBRISPEECH_REPO = "openslr/librispeech_asr"
FLEURS_REPO = "google/fleurs"

# --- SC-008 uniform decode/resample ---
# ONE shared resampler with fixed params for every clip (no per-corpus
# Audio(sampling_rate=...) cast, no sr==16000 special-casing). "soxr_hq" is
# librosa's default; pinning it explicitly makes the resampler version-independent
# and identical across all corpora, removing the class-correlated "was-librosa-
# resampled" artifact (together with SPEC-CHANGE-007). See SPEC-CHANGE-008.
RESAMPLE_RES_TYPE = "soxr_hq"

# --- SNR stratification buckets (dB) ---
SNR_EDGES = [15.0, 30.0]        # -> "low" (<15) / "mid" (15-30) / "high" (>30)

# --- paths ---
DATA_ROOT = Path(os.environ.get("MELGUARD_DATA", "/workspace/data"))
CLIPS_DIR = DATA_ROOT / "clips"
METADATA_DIR = DATA_ROOT / "metadata"


def snr_bucket(snr_db: float) -> str:
    if snr_db < SNR_EDGES[0]:
        return "low"
    if snr_db < SNR_EDGES[1]:
        return "mid"
    return "high"


def iter_metadata(stage: str | None = None):
    """Yield per-clip records from DATA_ROOT/metadata/*.jsonl (all, or one stage)."""
    if not METADATA_DIR.exists():
        return
    files = sorted(METADATA_DIR.glob(f"{stage}.jsonl" if stage else "*.jsonl"))
    for jf in files:
        with jf.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
