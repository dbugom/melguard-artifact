"""Loaders for the locked JSON manifests.

Every module that needs a locked quantity (sample counts, lambda_det values,
epsilon, thresholds, model pins, seeds) imports it from here so there is exactly
one source of truth on disk. Do not hardcode NUMBER_LOCK values elsewhere.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

# repo root = parent of the melguard/ package dir
ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / "env"

NUMBER_LOCK_PATH = ENV / "NUMBER_LOCK.lock.json"
MODEL_PINS_PATH = ENV / "MODEL_PINS.lock.json"
DATA_MANIFEST_PATH = ENV / "DATA_MANIFEST.lock.json"
GOVERNANCE_PATH = ENV / "GOVERNANCE.lock.json"


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=None)
def number_lock() -> dict:
    return _load(NUMBER_LOCK_PATH)


@lru_cache(maxsize=None)
def model_pins() -> dict:
    return _load(MODEL_PINS_PATH)


@lru_cache(maxsize=None)
def data_manifest() -> dict:
    return _load(DATA_MANIFEST_PATH)


@lru_cache(maxsize=None)
def governance() -> dict:
    return _load(GOVERNANCE_PATH)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file, matching `sha256sum` / coreutils."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# Convenience accessors used across the codebase -----------------------------

def lambda_det() -> list[float]:
    return list(number_lock()["lambda_det"])


def seeds() -> list[int]:
    return list(number_lock()["attack_seeds"])


def trigger_budget() -> dict:
    return dict(number_lock()["trigger_budget"])


def headline_bar() -> dict:
    return dict(number_lock()["headline_bar"])


def restricted_tools() -> list[str]:
    return list(number_lock()["restricted_tools"])


if __name__ == "__main__":  # quick manual dump
    nl = number_lock()
    print("lambda_det        :", nl["lambda_det"])
    print("held-out adv (N)  :", nl["held_out"]["adversarial_primary"])
    print("benign total      :", nl["held_out"]["benign_total"])
    print("trigger eps / L    :", nl["trigger_budget"]["linf_eps"], nl["trigger_budget"]["max_seconds"])
    print("SALMONN pin       :", model_pins()["models"]["salmonn"]["hf_repo"],
          "@", model_pins()["models"]["salmonn"]["hf_revision"])
