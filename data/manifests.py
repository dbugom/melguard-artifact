"""SHA-256 manifest engine for the data corpora.

Design (keeps the committed lock small while staying tamper-evident):
  - Per-clip records live in DATA_ROOT/metadata/*.jsonl (written by prepare.py).
  - DATA_ROOT/MANIFEST.sha256 lists "sha256  relpath" for every clip (coreutils
    format) — lives with the data on the volume/R2, not in git.
  - env/DATA_MANIFEST.lock.json stores only a SUMMARY: per-split count + a rollup
    SHA-256 (hash of that split's sorted "sha256 path" lines). Small, committable,
    and drift-detecting. The eval harness verifies clips against MANIFEST.sha256
    and the rollups against this lock; any drift aborts.

CLI:
  python data/manifests.py build     # (re)build MANIFEST.sha256 + update the lock summary
  python data/manifests.py verify    # recompute and compare; nonzero exit on drift
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from melguard.locks import ROOT, sha256_file  # noqa: E402

from data.config import DATA_ROOT, METADATA_DIR, iter_metadata  # noqa: E402

LOCK_PATH = ROOT / "env" / "DATA_MANIFEST.lock.json"
MANIFEST_SHA = DATA_ROOT / "MANIFEST.sha256"


def _rollup(lines: list[str]) -> str:
    """Deterministic rollup hash of sorted 'sha256  path' lines for a split."""
    h = hashlib.sha256()
    for line in sorted(lines):
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def build() -> dict:
    """Compute per-clip hashes, write MANIFEST.sha256, update the lock summary."""
    per_split: dict[str, list[str]] = defaultdict(list)
    total = 0
    all_lines: list[str] = []
    for rec in iter_metadata():
        rel = rec["path"]
        abspath = DATA_ROOT / rel
        if not abspath.exists():
            raise FileNotFoundError(f"metadata references missing clip: {rel}")
        digest = rec.get("sha256") or sha256_file(abspath)
        line = f"{digest}  {rel}"
        per_split[rec["role"]].append(line)
        all_lines.append(line)
        total += 1

    MANIFEST_SHA.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_SHA.write_text("\n".join(sorted(all_lines)) + "\n", encoding="utf-8")

    summary = {role: {"count": len(lines), "rollup_sha256": _rollup(lines)}
               for role, lines in sorted(per_split.items())}

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    lock["schema_version"] = 2
    lock["total_clips"] = total
    lock["manifest_sha256_file"] = "MANIFEST.sha256 (on DATA_ROOT / R2, not git)"
    lock["splits_summary"] = summary
    LOCK_PATH.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print(f"built manifest: {total} clips across {len(summary)} roles")
    for role, s in summary.items():
        print(f"  {role:16s} n={s['count']:6d}  rollup={s['rollup_sha256'][:12]}...")
    return summary


def verify() -> int:
    """Recompute rollups from metadata and compare to the committed lock. 0=ok."""
    if not LOCK_PATH.exists():
        print("no lock to verify"); return 1
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    expected = lock.get("splits_summary", {})
    if not expected:
        print("lock has no splits_summary — run 'build' first"); return 1

    per_split: dict[str, list[str]] = defaultdict(list)
    drift = []
    for rec in iter_metadata():
        rel = rec["path"]
        abspath = DATA_ROOT / rel
        if not abspath.exists():
            drift.append(f"missing clip: {rel}"); continue
        digest = sha256_file(abspath)  # recompute from disk (do not trust recorded)
        if rec.get("sha256") and rec["sha256"] != digest:
            drift.append(f"clip hash drift: {rel}")
        per_split[rec["role"]].append(f"{digest}  {rel}")

    for role, s in expected.items():
        got = _rollup(per_split.get(role, []))
        if got != s["rollup_sha256"]:
            drift.append(f"rollup drift in role '{role}': expected {s['rollup_sha256'][:12]} got {got[:12]}")
        if len(per_split.get(role, [])) != s["count"]:
            drift.append(f"count drift in role '{role}': expected {s['count']} got {len(per_split.get(role, []))}")

    if drift:
        print("MANIFEST VERIFY FAILED:")
        for d in drift[:20]:
            print("  -", d)
        return 2
    print(f"manifest verify OK ({sum(s['count'] for s in expected.values())} clips)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "verify"])
    args = ap.parse_args()
    if args.cmd == "build":
        build()
    else:
        raise SystemExit(verify())
