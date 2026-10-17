"""Phase-8 black-box model transfer to Qwen2-Audio-7B-Instruct (344-83, exploratory).

344-83: "black-box model transfer to Qwen2-Audio-7B-Instruct, reporting attack-success rate
only (not detector AUC, since the target model differs)". So this needs NEITHER the benign
clips NOR our detector — it only asks whether the dev-seed s31337 triggers, optimized as a
white-box attack on SALMONN, still drive a restricted-tool call on a DIFFERENT audio LLM.

Non-sealed by construction (dev-seed triggers), and there is no held-out data here at all.
SC-011: the pinned 7B weights were never GPU load-tested before Phase 8; this is that test.
The model is loaded in 4-bit NF4 (same quantization family as our SALMONN) to fit the 4090
alongside nothing else. Attack-success = the emitted text parses to the trigger's target
restricted tool call, judged by the SAME harness predicate used for SALMONN (harness.actions).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.actions import l_actuate, parse_salmonn_output  # noqa: E402
from attack.surrogate import INSTRUCTION, TARGETS  # noqa: E402
from melguard.locks import number_lock  # noqa: E402

_NL = number_lock()
SR = 16000
DEV_SEED = 31337
TAG_BY_LAMBDA = {0.0: "phase5_lam0", 1.0: "phase6_lam1"}
TOOLS = ("transfer_money", "ssh_exec", "open_door")
LANGS = ("en", "zh-CN", "es")
BACKUP = Path("/workspace/melguard-backup")


def _pins() -> tuple[str, str]:
    m = _NL if False else json.loads(Path("env/MODEL_PINS.lock.json").read_text())
    p = m["models"]["transfer_exploratory"]
    return p["hf_repo"], p["hf_revision"]


def _load_dev_triggers() -> list[dict]:
    out = []
    for lam, tag in TAG_BY_LAMBDA.items():
        sep = f"_s{DEV_SEED}_"
        for fp in sorted(BACKUP.glob(f"triggers/{tag}_*{sep}*.flac")):
            stem = fp.stem
            left, uid = stem.split(sep, 1)
            rest = left[len(tag) + 1:]
            tool = next((t for t in TOOLS if rest.startswith(t + "_")), None)
            if tool is None:
                continue
            lang = rest[len(tool) + 1:]
            if lang not in LANGS:
                continue
            out.append({"uid": stem, "path": str(fp), "language": lang, "tool": tool, "lam": lam})
    return out


def _load_qwen(device: str):
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2AudioForConditionalGeneration)
    repo, rev = _pins()
    proc = AutoProcessor.from_pretrained(repo, revision=rev)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        repo, revision=rev, quantization_config=bnb, device_map=device)
    model.eval()
    return model, proc


def _qwen_reply(model, proc, wav: np.ndarray, device: str, max_new_tokens: int = 64) -> str:
    conv = [{"role": "user", "content": [
        {"type": "audio", "audio_url": "trigger.wav"},
        {"type": "text", "text": INSTRUCTION}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=text, audios=[wav], sampling_rate=SR, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             num_beams=1)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return proc.batch_decode(gen, skip_special_tokens=True)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="out/phase8")
    ap.add_argument("--limit", type=int, default=None, help="smoke: cap #triggers")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    triggers = _load_dev_triggers()
    if args.limit:
        triggers = triggers[: args.limit]
    print(f"[qwen] loading Qwen2-Audio-7B-Instruct (4-bit)...", flush=True)
    model, proc = _load_qwen(args.device)
    peak = torch.cuda.max_memory_reserved() / 1024 ** 3
    print(f"[qwen] loaded; peak reserved so far {peak:.2f} GiB", flush=True)

    rows = []
    for i, r in enumerate(triggers):
        wav, sr = sf.read(r["path"], dtype="float32")
        if wav.ndim == 2:
            wav = wav[:, 0]
        if sr != SR:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR, res_type="fft")
        text = _qwen_reply(model, proc, wav, args.device)
        parsed = parse_salmonn_output(text)
        la = int(l_actuate(parsed, TARGETS[r["tool"]].spec()))
        rows.append({"lam": r["lam"], "language": r["language"], "tool": r["tool"],
                     "l_actuate": la, "action": parsed.action.value,
                     # attack-success is the STRICT targeted predicate (l_actuate); this
                     # separate flag captures whether the trigger induced ANY restricted
                     # tool call, which distinguishes "no transfer" from "transfers as
                     # generic restricted tool-calling but not as the targeted action".
                     "any_restricted": int(bool(parsed.is_restricted)),
                     "reply_head": text[:80]})
        if (i + 1) % 25 == 0:
            print(f"[qwen] {i+1}/{len(triggers)}  running actuation="
                  f"{np.mean([x['l_actuate'] for x in rows]):.3f}", flush=True)

    def _rate(sel):
        s = [x for x in rows if sel(x)]
        return {"n": len(s),
                "attack_success_rate": round(np.mean([x["l_actuate"] for x in s]), 6) if s else None,
                "any_restricted_toolcall_rate": round(np.mean([x["any_restricted"] for x in s]), 6)
                if s else None}

    peak = torch.cuda.max_memory_reserved() / 1024 ** 3
    result = {
        "study": "qwen2audio_transfer", "phase": 8, "confirmatory": False,
        "target_model": "Qwen/Qwen2-Audio-7B-Instruct", "revision": _pins()[1],
        "note": "EXPLORATORY black-box transfer (344-83): attack-success rate only, NOT "
                "detector AUC (the target model differs). Dev-seed s31337 triggers optimized "
                "white-box on SALMONN. No held-out or benign data involved. 4-bit NF4.",
        "peak_vram_gib": round(peak, 2), "n_triggers": len(rows),
        "overall": _rate(lambda x: True),
        "by_lambda": {str(l): _rate(lambda x, l=l: x["lam"] == l) for l in TAG_BY_LAMBDA},
        "by_tool": {t: _rate(lambda x, t=t: x["tool"] == t) for t in TOOLS},
        "by_language": {lg: _rate(lambda x, lg=lg: x["language"] == lg) for lg in LANGS},
        "action_histogram": {a: sum(1 for x in rows if x["action"] == a)
                             for a in sorted({x["action"] for x in rows})},
    }
    res = out_dir / "qwen2audio_transfer.json"
    res.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[qwen] DONE peak={peak:.2f} GiB  overall attack-success="
          f"{result['overall']['attack_success_rate']}  by_lambda="
          f"{ {k: v['attack_success_rate'] for k, v in result['by_lambda'].items()} }")
    print(f"  action histogram: {result['action_histogram']}")
    print(f"  wrote {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
