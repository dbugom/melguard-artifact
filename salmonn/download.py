"""Fetch + verify the pinned SALMONN-13B components (reads env/MODEL_PINS.lock.json).

Idempotent — resolves from the HF cache (HF_HOME on the volume) and only fetches
what is missing. Verifies BEATs + the SALMONN checkpoint against their locked
SHA-256, and downloads **only** the file patterns we need (safetensors for
Whisper; the .bin shards for Vicuna) so we never pull the tf/flax/bin dupes.

Usage:  python -m salmonn.download            # ensure + verify, print resolved paths
"""
from __future__ import annotations

import json

from melguard.locks import model_pins, sha256_file

# Only the files SALMONN actually loads (avoids ~18 GB of duplicate whisper formats).
_WHISPER_PATTERNS = [
    "config.json", "generation_config.json", "model.safetensors", "merges.txt",
    "normalizer.json", "preprocessor_config.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "added_tokens.json", "special_tokens_map.json",
]
_VICUNA_PATTERNS = ["*.json", "*.model", "pytorch_model*.bin"]


def _pins() -> dict:
    return model_pins()["models"]["salmonn"]


def _verify(path: str, expected: str, name: str) -> None:
    got = sha256_file(path)
    if got != expected:
        raise RuntimeError(
            f"GUARDRAIL: {name} SHA-256 mismatch — got {got}, pinned {expected} ({path})"
        )
    print(f"[ok] {name} SHA-256 verified {got[:16]}…")


def ensure_components(verify: bool = True) -> dict:
    """Return resolved on-disk paths for the four SALMONN components."""
    from huggingface_hub import snapshot_download, hf_hub_download

    p = _pins()
    c = p["components"]

    whisper_dir = snapshot_download(
        c["whisper"]["hf_repo"], revision=c["whisper"]["hf_revision"], allow_patterns=_WHISPER_PATTERNS
    )
    vicuna_dir = snapshot_download(
        c["vicuna"]["hf_repo"], revision=c["vicuna"]["hf_revision"], allow_patterns=_VICUNA_PATTERNS
    )
    beats_file = hf_hub_download(
        c["beats"]["hf_repo"], c["beats"]["file"], revision=c["beats"]["hf_revision"]
    )
    ckpt_file = hf_hub_download(p["hf_repo"], p["ckpt_file"], revision=p["hf_revision"])

    if verify:
        _verify(beats_file, c["beats"]["sha256"], "BEATs")
        _verify(ckpt_file, p["ckpt_sha256"], "salmonn_v1.pth")

    return {
        "whisper_dir": whisper_dir,
        "vicuna_dir": vicuna_dir,
        "beats_file": beats_file,
        "ckpt_file": ckpt_file,
    }


if __name__ == "__main__":
    paths = ensure_components(verify=True)
    print(json.dumps(paths, indent=2))
