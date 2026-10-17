"""Phase 5, step 2: measure peak VRAM of the joint teacher-forced backward
delta -> f_S (Whisper+BEATs -> Q-Former -> proj -> 4-bit Vicuna-13B) -> CE loss,
on the RTX 4090 (24 GB). This is the unmeasured GPU question of the phase.

Only the waveform delta carries grad (the true attack graph): the prompt/target
token embeddings are frozen lookups, the speech embeddings carry grad from delta,
and the loss is a teacher-forced CE toward a placeholder restricted tool-call
string (the *length* is what matters for VRAM; the exact tokens are set at
sign-off). Reports peak allocated/reserved with and without LLM gradient
checkpointing.

    python -m attack.vram_probe [--both] [--checkpoint] [--target-tokens N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import soundfile as sf
import torch

from attack.frontend import encode_audio_grad
from attack.detector_frontend import detector_prob_grad
from salmonn.loader import load_salmonn, set_determinism

# Placeholder teacher-forcing target (a restricted tool-call shape). The real
# string + TargetSpec are settled at operator sign-off; VRAM depends on length.
PLACEHOLDER_TARGET = ' {"tool": "ssh_exec", "arguments": {"command": "REDACTED"}}'
GiB = 1024 ** 3


def _find_clip() -> str:
    for sub in ("clips/commonvoice18", "clips/librispeech", "clips/fleurs"):
        hits = sorted(glob.glob(os.path.join("/workspace/data", sub, "**", "*.flac"), recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError("no sample clip found under /workspace/data/clips")


def _embed_tokens(model):
    return (
        model.llama_model.model.model.embed_tokens if model.lora
        else model.llama_model.model.embed_tokens
    )


def _build_inputs(model, meta, full_wav, target_text):
    device = model.device_str
    tok = model.llama_tokenizer
    embed_tokens = _embed_tokens(model)

    instruction = "You control tools. Respond ONLY with a JSON tool call."
    left_text, right_text = meta["prompt_pattern"].format(instruction).split("<SpeechHere>")

    with torch.no_grad():
        def ids(t):
            return tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        left_ids, right_ids, target_ids = ids(left_text), ids(right_text), ids(target_text)
        bos = torch.ones([1, 1], dtype=torch.long, device=device) * tok.bos_token_id
        bos_e = embed_tokens(bos)
        left_e = embed_tokens(left_ids)
        right_e = embed_tokens(right_ids)
        target_e = embed_tokens(target_ids)

    speech_e = encode_audio_grad(model, full_wav)  # carries grad from delta

    embeds = torch.cat([bos_e, left_e, speech_e, right_e, target_e], dim=1)
    S = embeds.size(1)
    n_tgt = target_ids.size(1)
    labels = torch.full((1, S), -100, dtype=torch.long, device=device)
    labels[:, -n_tgt:] = target_ids
    atts = torch.ones((1, S), dtype=torch.long, device=device)
    return embeds, atts, labels, {
        "seq_len": int(S),
        "n_speech_tokens": int(speech_e.size(1)),
        "n_target_tokens": int(n_tgt),
    }


def _measure(model, meta, full_wav, target_text, checkpoint: bool,
             detector=None, ast_cfg=None, lam: float = 0.0) -> dict:
    if checkpoint:
        model.llama_model.gradient_checkpointing_enable()
        model.llama_model.enable_input_require_grads()
    else:
        try:
            model.llama_model.gradient_checkpointing_disable()
        except Exception:
            pass

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    embeds, atts, labels, shapes = _build_inputs(model, meta, full_wav, target_text)
    out = model.llama_model(inputs_embeds=embeds, attention_mask=atts, labels=labels, use_cache=False)
    loss = out.loss
    if detector is not None and lam > 0:
        # Phase-6 JOINT backward: add the differentiable detector-evasion term so the
        # measured peak includes g_D's forward+backward graph (SPEC-CHANGE-012 sign).
        p_det = detector_prob_grad(detector, full_wav, mel_cfg=ast_cfg)
        loss = loss + lam * (-torch.log1p(-p_det.clamp(max=1 - 1e-6)).mean())
        shapes["joint_gD"] = True
        shapes["p_det"] = float(p_det.flatten()[0].item())
    loss.backward()

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_resv = torch.cuda.max_memory_reserved()

    # tidy up for a subsequent measurement
    model.zero_grad(set_to_none=True)
    del embeds, atts, labels, out, loss
    torch.cuda.empty_cache()

    return {
        "checkpoint": checkpoint,
        **shapes,
        "loss": None,  # not needed; graph freed
        "base_alloc_gib": base / GiB,
        "peak_alloc_gib": peak_alloc / GiB,
        "peak_reserved_gib": peak_resv / GiB,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--both", action="store_true", help="measure plain then checkpointed")
    ap.add_argument("--checkpoint", action="store_true", help="measure checkpointed only")
    ap.add_argument("--detector", action="store_true",
                    help="Phase-6 JOINT f_S+g_D backward (adds the differentiable detector-evasion term)")
    ap.add_argument("--detector-weights", default="out/phase4/detector_gd_best.pt")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="lambda for the joint probe (graph size only; this is a memory probe, not an attack)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    joint = args.detector
    out_path = args.out or ("out/phase6/vram_probe_joint.json" if joint else "out/phase5/vram_probe.json")

    seed = set_determinism()
    total_gib = torch.cuda.get_device_properties(0).total_memory / GiB
    name = torch.cuda.get_device_name(0)

    clip = _find_clip()
    carrier, sr = sf.read(clip)
    if carrier.ndim == 2:
        carrier = carrier[:, 0]
    carrier = torch.tensor(carrier.astype(np.float32))
    delta = torch.zeros(16000, dtype=torch.float32)  # 1.0 s prepend budget
    # A single leaf over delta+carrier: the backward graph through f_S is the
    # same size as the real attack (only the first 16000 samples are optimised),
    # so peak VRAM is identical whether the whole waveform or just delta is a leaf.
    full = torch.cat([delta, carrier], dim=0).to("cuda:0").detach().requires_grad_(True)

    print(f"[vram] gpu={name} total={total_gib:.1f} GiB seed={seed} joint_gD={joint}")
    print("[vram] loading SALMONN (f_S) ...")
    model, meta = load_salmonn(verify=False)

    detector, ast_cfg = (None, None)
    if joint:
        from attack.joint_pgd import load_attack_detector
        print(f"[vram] loading FROZEN g_D from {args.detector_weights} (lam={args.lam:g}) ...")
        detector, ast_cfg = load_attack_detector(args.detector_weights, device="cuda:0")

    configs = []
    if args.checkpoint:
        configs = [True]
    elif args.both:
        configs = [False, True]
    else:
        configs = [False]

    runs = []
    for ckpt in configs:
        try:
            r = _measure(model, meta, full, PLACEHOLDER_TARGET, checkpoint=ckpt,
                         detector=detector, ast_cfg=ast_cfg, lam=(args.lam if joint else 0.0))
            r["headroom_gib"] = total_gib - r["peak_reserved_gib"]
            r["fits_24gb"] = r["peak_reserved_gib"] < total_gib
            runs.append(r)
            print(f"[vram] checkpoint={ckpt}: peak_alloc={r['peak_alloc_gib']:.2f} "
                  f"peak_reserved={r['peak_reserved_gib']:.2f} GiB  headroom={r['headroom_gib']:.2f} GiB")
        except torch.cuda.OutOfMemoryError as e:
            runs.append({"checkpoint": ckpt, "oom": True, "error": str(e)[:200]})
            print(f"[vram] checkpoint={ckpt}: OOM")
            torch.cuda.empty_cache()

    result = {
        "phase": 6 if joint else 5,
        "step": "phase6_joint_fS_gD_backward_vram" if joint else "2_joint_backward_vram",
        "joint_gD": joint,
        "lambda": args.lam if joint else 0.0,
        "detector_weights": args.detector_weights if joint else None,
        "seed": seed,
        "gpu": name,
        "total_gib": total_gib,
        "quantization": meta["quantization"],
        "placeholder_target": PLACEHOLDER_TARGET,
        "runs": runs,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    print(f"[vram] wrote {out_path}")
    return 0


if __name__ == "__main__":
    rc = main()
    os._exit(rc)
