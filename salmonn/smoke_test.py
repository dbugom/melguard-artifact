"""Phase-2 acceptance gate: forward-only SALMONN-13B smoke test.

Loads the pinned 4-bit SALMONN-13B, runs **deterministic greedy** generation on a
single benign clip twice, and checks:
  * the model assembles + the trained delta loads (no unexpected ckpt keys),
  * it fits in the 24 GB RTX 4090 (no OOM; peak VRAM reported),
  * the output is reproducible (two greedy runs are byte-identical).

Emits ``out/phase2/smoke_result.json`` (+ a SHA-256 in ``out/phase2/MANIFEST.sha256``).
No waveform is written — only the clip's path + SHA-256, the text response, VRAM,
and the config. Guardrail-clean: benign clip in, text out, no triggers.

Usage:
  python -m salmonn.smoke_test [--wav PATH] [--data-root /workspace/data]
                               [--device cuda:0] [--prompt "..."]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from melguard.locks import MODEL_PINS_PATH, model_pins, sha256_file


def _pick_default_clip(data_root: Path) -> Path:
    """Deterministically pick the lexicographically-first benign wav/flac clip."""
    cands = sorted(
        p for ext in ("*.wav", "*.flac")
        for p in data_root.rglob(ext)
    )
    if not cands:
        raise SystemExit(
            f"No .wav/.flac clip found under {data_root}; pass --wav explicitly."
        )
    return cands[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, default=None)
    ap.add_argument("--data-root", type=str, default="/workspace/data")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--prompt", type=str, default="Please transcribe the speech.")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    import torch
    from salmonn.loader import load_salmonn, set_determinism

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out) if args.out else repo_root / "out" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)

    wav = Path(args.wav) if args.wav else _pick_default_clip(Path(args.data_root))
    print(f"== Phase-2 SALMONN smoke ==\nclip: {wav}\nprompt: {args.prompt!r}")

    seed = set_determinism()
    torch.cuda.reset_peak_memory_stats()

    model, meta = load_salmonn(device=args.device, verify=True)

    # deterministic greedy generation, run twice for reproducibility
    out1 = model.generate(str(wav), prompt=args.prompt, deterministic=True)[0]
    out2 = model.generate(str(wav), prompt=args.prompt, deterministic=True)[0]
    reproducible = (out1 == out2)

    # The salmonn_v1.pth checkpoint (transformers 4.28 era) carries per-layer
    # `self_attn.rotary_emb.inv_freq` buffers. transformers 4.46 moved rotary
    # embeddings to a single model-level module and recomputes inv_freq from the
    # config, so these appear as "unexpected" on load. They are non-trained
    # constants — benign. Any *other* unexpected key would signal a real mismatch.
    unexpected = list(model._ckpt_unexpected)
    real_unexpected = [k for k in unexpected if not k.endswith("rotary_emb.inv_freq")]

    peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
    peak_resv = torch.cuda.max_memory_reserved() / (1024**2)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)

    result = {
        "phase": 2,
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_mib": round(total_vram, 1),
        "peak_alloc_mib": round(peak_alloc, 1),
        "peak_reserved_mib": round(peak_resv, 1),
        "fits_24gb": peak_resv < total_vram,
        "quantization": meta["quantization"],
        "seed": seed,
        "clip_path": str(wav),
        "clip_sha256": sha256_file(wav),
        "prompt": args.prompt,
        "response": out1,
        "response_reproducible": reproducible,
        "ckpt_unexpected_real": real_unexpected,
        "ckpt_unexpected_benign_rotary": len(unexpected) - len(real_unexpected),
        "ckpt_missing_key_count": model._ckpt_missing,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "model_pins_sha": sha256_file(MODEL_PINS_PATH)[:16],
    }

    res_path = out_dir / "smoke_result.json"

    ok = (
        result["fits_24gb"]
        and reproducible
        and len(real_unexpected) == 0
        and len(out1.strip()) > 0
    )
    result["gate_pass"] = ok
    res_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "MANIFEST.sha256").write_text(
        f"{sha256_file(res_path)}  smoke_result.json\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    print("\n--- PHASE-2 GATE:", "PASS ✅" if ok else "FAIL ❌", "---")
    print(f"peak reserved {peak_resv:.0f} / {total_vram:.0f} MiB | "
          f"reproducible={reproducible} | real_unexpected_keys={len(real_unexpected)} "
          f"(+{len(unexpected) - len(real_unexpected)} benign rotary) | "
          f"response_len={len(out1.strip())}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
