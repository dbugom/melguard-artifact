"""Joint sign-PGD attack on delta (spec §5.4 / NUMBER_LOCK).

    minimise_delta   CE( target_tool_call | (delta (+) carrier), instruction )
                     [ - lambda_det * log(1 - p_det)  ]   <-- lambda>0 in Phase 6
    subject to       ||delta||_inf <= eps (0.04),  delta is a <=1.0 s prepend,
                     50 ms linear cross-fade into the carrier.

Phase 5 runs at **lambda = 0** (no detector term): a pure actuation attack that
measures how strong the trigger is before the detector is in the loop. lambda>0
(Phase 6) adds the differentiable detector-evasion term (``attack/detector_frontend``)
against the FROZEN Phase-4 g_D. Per **SPEC-CHANGE-012** the term is the
evasion-correct sign: the loop MINIMISES ``CE - lambda*log(1 - p_det)`` so p_det
DROPS as steps progress (the literal spec maximise-form ``- lambda*log(1-g_D)`` is
anti-evasion — it would push p_det UP). Generating any lambda>0 trigger STARTS the
90-day coordinated-disclosure clock, so it is gated behind an explicit operator
acknowledgement (``--start-disclosure-clock`` / ``disclosure_ack=True``).

Locked knobs come from NUMBER_LOCK (steps=5000, checkpoint_every=500,
sign_gradient_linf_clip; eps=0.04, n_samples=16000, crossfade_ms=50, prepend).
Per SC-009 the optimised trigger waveform + payload are written to the PRIVATE
``melguard-backup/{triggers,payloads}`` (+ R2), SHA-256 manifested, NEVER git.
The git-tracked JSONL log carries params + metrics + a <=80-char decoded head
only (guardrail: no full payload, no waveform).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from attack.surrogate import TARGETS, Target, measured_actuation, teacher_forced_ce
from attack.frontend import whisper_mel_config
from attack.detector_frontend import ast_mel_config, detector_prob_grad
from melguard.locks import number_lock
from salmonn.loader import load_salmonn, set_determinism

SR = 16000
BACKUP_ROOT = Path(os.environ.get("MELGUARD_BACKUP", "/workspace/melguard-backup"))
DISCLOSURE_TIMER = Path("out/disclosure-timer.md")


# ------------------------------------------------------------------ Phase-6 detector (g_D) + disclosure gate
def load_attack_detector(weights: str = "out/phase4/detector_gd_best.pt", device: str = "cuda:0"):
    """Load the FROZEN Phase-4 detector g_D + its pinned AST mel config for the
    lambda>0 evasion term. g_D is NOT trained here (frozen, eval); the adversarial
    retrain of g_D is a separate Phase-6 step. Returns ``(detector, ast_mel_cfg)``."""
    from transformers import ASTFeatureExtractor
    from detector.model import MelGuardDetector, detector_pins
    det = MelGuardDetector()
    sd = torch.load(weights, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    det.load_state_dict(sd, strict=False)
    det.eval().to(device)
    for p in det.parameters():
        p.requires_grad_(False)
    d = detector_pins()
    fe = ASTFeatureExtractor.from_pretrained(d["hf_repo"], revision=d["hf_revision"])
    return det, ast_mel_config(fe)


def evasion_penalty(p_det: torch.Tensor) -> torch.Tensor:
    """SPEC-CHANGE-012 evasion term: ``-log(1 - p_det)`` (>=0, ->inf as p_det->1).
    Added to the MINIMISED loss, so descent drives p_det DOWN (below tau_op). The
    clamp keeps the log finite when a trigger is well-detected (p_det ~ 1)."""
    return -torch.log1p(-p_det.clamp(max=1 - 1e-6))


def _assert_disclosure_ack(lam: float, ack: bool) -> None:
    """lambda>0 optimises a detector-aware adaptive trigger, which STARTS the
    90-day coordinated-disclosure clock (CLAUDE.md guardrail). Refuse to generate
    unless the operator has explicitly acknowledged disclosure logistics."""
    if lam > 0 and not ack:
        raise SystemExit(
            "GUARDRAIL: lambda>0 generates a detector-aware adaptive trigger, which "
            "starts the 90-day disclosure clock. Re-run with --start-disclosure-clock "
            "(disclosure_ack=True) only AFTER the operator has confirmed disclosure "
            "logistics. (VRAM/equivalence probes are non-generating and exempt.)"
        )

# i%3 tool rotation across the 25 carriers/lang (operator-ratified 2026-07-14):
# every (lambda, seed, lang) cell covers all 3 tools, cube stays at NUMBER_LOCK 900.
TOOLS_ORDER = ["transfer_money", "ssh_exec", "open_door"]


def _sample_seed(seed: int, uid: str) -> int:
    import hashlib
    return int(hashlib.sha256(f"{seed}:{uid}".encode()).hexdigest()[:8], 16)


def _lam_tag(lam: float) -> str:
    """Output tag/dir stem: lambda=0 keeps the Phase-5 name; lambda>0 is Phase 6."""
    return "phase5_lam0" if lam == 0 else f"phase6_lam{lam:g}"


# ------------------------------------------------------------------ trigger construction
def prepend_crossfade(delta: torch.Tensor, carrier: torch.Tensor, crossfade_samples: int) -> torch.Tensor:
    """x = [ trigger head | (trigger tail x fade-out + carrier head x fade-in) | carrier tail ].

    Differentiable w.r.t. ``delta``; ``carrier`` is constant. Linear 50 ms fade."""
    cf = crossfade_samples
    dev, dt = delta.device, delta.dtype
    if cf <= 0:
        return torch.cat([delta, carrier], dim=0)
    head = delta[:-cf]
    d_tail = delta[-cf:]
    c_head = carrier[:cf]
    c_tail = carrier[cf:]
    fade_out = torch.linspace(1.0, 0.0, cf, device=dev, dtype=dt)
    fade_in = torch.linspace(0.0, 1.0, cf, device=dev, dtype=dt)
    mid = d_tail * fade_out + c_head * fade_in
    return torch.cat([head, mid, c_tail], dim=0)


def prepend_crossfade_batched(delta: torch.Tensor, carriers: torch.Tensor, crossfade_samples: int) -> torch.Tensor:
    """Batched prepend+crossfade. ``delta`` (B, D), ``carriers`` (B, C) (equal C),
    -> (B, D + C - cf). Differentiable w.r.t. ``delta``."""
    cf = crossfade_samples
    if cf <= 0:
        return torch.cat([delta, carriers], dim=1)
    head = delta[:, :-cf]
    d_tail = delta[:, -cf:]
    c_head = carriers[:, :cf]
    c_tail = carriers[:, cf:]
    fade_out = torch.linspace(1.0, 0.0, cf, device=delta.device, dtype=delta.dtype)
    fade_in = torch.linspace(0.0, 1.0, cf, device=delta.device, dtype=delta.dtype)
    mid = d_tail * fade_out + c_head * fade_in
    return torch.cat([head, mid, c_tail], dim=1)


def standardize_carrier(wav: torch.Tensor, target_samples: int) -> torch.Tensor:
    """Pad (silence) or truncate a carrier to a fixed length so a batch stacks and
    the optimisation / measurement / saved-trigger representations all match."""
    n = wav.shape[-1]
    if n < target_samples:
        return torch.nn.functional.pad(wav, (0, target_samples - n))
    return wav[..., :target_samples]


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ attack loop
def run_attack(
    model, meta, carrier: torch.Tensor, carrier_meta: dict, target: Target, *,
    seed: int, lam: float = 0.0, steps: int = 5000, ckpt_every: int = 500,
    eps: float = 0.04, delta_len: int = 16000, crossfade_ms: int = 50,
    alpha: float = 0.002, alpha_final: float | None = None,
    detector=None, ast_cfg=None, disclosure_ack: bool = False,
    log_jsonl: Path | None = None, measure_actuation: bool = True,
    tag: str = "phase5_lam0",
) -> dict:
    if lam > 0:
        _assert_disclosure_ack(lam, disclosure_ack)
        assert detector is not None and ast_cfg is not None, "lambda>0 needs the frozen detector g_D + ast_cfg"
    device = model.device_str
    mel_cfg = whisper_mel_config(model, device=device)
    cf = int(round(crossfade_ms / 1000.0 * SR))

    carrier = carrier.to(device=device, dtype=torch.float32).detach()
    g = torch.Generator(device="cpu").manual_seed(seed)
    # brief §5.4: initialise delta with white Gaussian x eps, then project to the ball.
    delta0 = (torch.randn(delta_len, generator=g) * eps).clamp(-eps, eps)
    delta = delta0.to(device=device, dtype=torch.float32).clone().requires_grad_(True)

    run_id = f"{tag}_{target.tool}_{carrier_meta.get('language','na')}_s{seed}_{carrier_meta.get('uid','na')}"
    ckpts: list[dict] = []
    t0 = time.time()
    log_fh = open(log_jsonl, "a", encoding="utf-8") if log_jsonl else None

    def checkpoint(step: int, ce_val: float):
        with torch.no_grad():
            x = prepend_crossfade(delta, carrier, cf).clamp(-1, 1)
            linf = delta.abs().max().item()
            l2 = delta.norm().item()
            pdet = (float(detector_prob_grad(detector, x, mel_cfg=ast_cfg).flatten()[0].item())
                    if detector is not None else None)
        rec = {
            "run_id": run_id, "tool": target.tool, "language": carrier_meta.get("language"),
            "seed": seed, "lambda": lam, "step": step, "ce_loss": ce_val,
            "target_prob": math.exp(-ce_val), "delta_linf": linf, "delta_l2": l2,
            "eps": eps, "within_eps": linf <= eps + 1e-6,
            "elapsed_s": round(time.time() - t0, 2),
        }
        if pdet is not None:
            rec["p_det"] = pdet          # Phase-6: detector score; should DROP over steps
        if measure_actuation:
            rec.update({f"act_{k}": v for k, v in measured_actuation(model, meta, x, target).items()})
        ckpts.append(rec)
        if log_fh:
            log_fh.write(json.dumps(rec) + "\n"); log_fh.flush()
        act = rec.get("act_l_actuate", "-")
        pstr = f" p_det={pdet:.4f}" if pdet is not None else ""
        print(f"[pgd] {run_id} step {step:4d}/{steps} ce={ce_val:.4f} "
              f"p={rec['target_prob']:.3f} linf={rec['delta_linf']:.4f}{pstr} act={act}", flush=True)

    # initial state (step 0)
    with torch.no_grad():
        x0 = prepend_crossfade(delta, carrier, cf).clamp(-1, 1)
        ce0 = teacher_forced_ce(model, meta, x0, target.target_text, mel_cfg=mel_cfg).item()
    checkpoint(0, ce0)

    for step in range(1, steps + 1):
        x = prepend_crossfade(delta, carrier, cf).clamp(-1, 1)
        ce = teacher_forced_ce(model, meta, x, target.target_text, mel_cfg=mel_cfg)
        if lam > 0:
            # SPEC-CHANGE-012 evasion-correct objective: minimise CE - lambda*log(1 - p_det)
            # so descent drives p_det DOWN (below tau_op) while CE keeps falling (actuation).
            p_det = detector_prob_grad(detector, x, mel_cfg=ast_cfg)
            loss = ce + lam * evasion_penalty(p_det).mean()
        else:
            loss = ce
        (grad,) = torch.autograd.grad(loss, delta)
        a_step = alpha if alpha_final is None else (
            alpha_final + 0.5 * (alpha - alpha_final) * (1 + math.cos(math.pi * step / steps)))
        with torch.no_grad():
            # brief §5.4: delta <- clip(delta + alpha*sign(grad L), eps). L = -loss, so
            # sign(grad L) = -sign(grad loss); minimising the loss maximises actuation
            # (and, for lambda>0, drives p_det down).
            delta.add_(-a_step * grad.sign())
            delta.clamp_(-eps, eps)            # project onto the l_inf ball
        if step % ckpt_every == 0 or step == steps:
            checkpoint(step, ce.item())

    if log_fh:
        log_fh.close()
    wall = time.time() - t0
    summary = {
        "run_id": run_id, "tool": target.tool, "language": carrier_meta.get("language"),
        "carrier_uid": carrier_meta.get("uid"), "seed": seed, "lambda": lam,
        "steps": steps, "alpha": alpha, "eps": eps, "crossfade_ms": crossfade_ms,
        "delta_len": delta_len, "final": ckpts[-1], "init": ckpts[0],
        "wall_s": round(wall, 1), "sec_per_step": round(wall / max(1, steps), 4),
        "checkpoints": ckpts,
    }
    # persist the optimised trigger (SC-009: private backup only, never git)
    summary["backup"] = _write_backup(delta.detach(), carrier, cf, target, carrier_meta, summary)
    return summary


# ------------------------------------------------------------------ batched attack loop
def run_attack_batched(
    model, meta, carriers: list, carrier_metas: list, targets: list, *,
    seed: int, lam: float = 0.0, steps: int = 5000, ckpt_every: int = 500,
    eps: float = 0.04, delta_len: int = 16000, crossfade_ms: int = 50,
    alpha: float = 0.002, alpha_final: float | None = None, carrier_seconds: float = 10.0,
    detector=None, ast_cfg=None, disclosure_ack: bool = False,
    log_jsonl: Path | None = None, measure_actuation: bool = True,
    tag: str = "phase5_lam0", backup: bool = True,
) -> list[dict]:
    """Optimise B triggers in one batched joint backward (the 4090 throughput path
    the operator approved). Each (carrier, target) is an independent attack sharing
    only the batched forward; sign-PGD is per-sample correct despite the mean loss.

    If ``alpha_final`` is set, the step size cosine-anneals ``alpha`` -> ``alpha_final``
    over the run (large early exploration, small late refinement to lock in the
    greedy argmax); otherwise a constant ``alpha`` is used."""
    from attack.surrogate import teacher_forced_ce, teacher_forced_ce_batched

    if lam > 0:
        _assert_disclosure_ack(lam, disclosure_ack)
        assert detector is not None and ast_cfg is not None, "lambda>0 needs the frozen detector g_D + ast_cfg"
    device = model.device_str
    mel_cfg = whisper_mel_config(model, device=device)
    cf = int(round(crossfade_ms / 1000.0 * SR))
    L_c = int(round(carrier_seconds * SR))
    B = len(carriers)

    car = torch.stack([standardize_carrier(c.to(device, torch.float32), L_c) for c in carriers], dim=0)
    # per-(seed, carrier) reproducible Gaussian init (brief §5.4) — independent of
    # batch grouping so a trigger is identical however carriers are batched.
    rows = []
    for b in range(B):
        gb = torch.Generator(device="cpu").manual_seed(_sample_seed(seed, carrier_metas[b].get("uid", "na")))
        rows.append((torch.randn(delta_len, generator=gb) * eps).clamp(-eps, eps))
    delta = torch.stack(rows, 0).to(device, torch.float32).requires_grad_(True)
    tgt_texts = [t.target_text for t in targets]
    run_ids = [f"{tag}_{targets[b].tool}_{carrier_metas[b].get('language','na')}_s{seed}_{carrier_metas[b].get('uid','na')}"
               for b in range(B)]

    log_fh = open(log_jsonl, "a", encoding="utf-8") if log_jsonl else None
    t0 = time.time()

    def checkpoint(step: int):
        with torch.no_grad():
            x = prepend_crossfade_batched(delta, car, cf).clamp(-1, 1)
            pdet_all = (detector_prob_grad(detector, x, mel_cfg=ast_cfg).flatten().tolist()
                        if detector is not None else None)
            for b in range(B):
                xb = x[b]
                ce_b = teacher_forced_ce(model, meta, xb, tgt_texts[b], mel_cfg=mel_cfg).item()
                rec = {
                    "run_id": run_ids[b], "tool": targets[b].tool,
                    "language": carrier_metas[b].get("language"), "carrier_uid": carrier_metas[b].get("uid"),
                    "seed": seed, "lambda": lam, "step": step, "ce_loss": ce_b,
                    "target_prob": math.exp(-ce_b), "delta_linf": delta[b].abs().max().item(),
                    "delta_l2": delta[b].norm().item(), "eps": eps,
                    "within_eps": delta[b].abs().max().item() <= eps + 1e-6,
                    "elapsed_s": round(time.time() - t0, 2),
                }
                if pdet_all is not None:
                    rec["p_det"] = pdet_all[b]      # Phase-6: should DROP over steps
                if measure_actuation:
                    rec.update({f"act_{k}": v for k, v in measured_actuation(model, meta, xb, targets[b]).items()})
                if log_fh:
                    log_fh.write(json.dumps(rec) + "\n")
            if log_fh:
                log_fh.flush()
        print(f"[pgd-batch] step {step:4d}/{steps} B={B} linf_max={delta.abs().max().item():.4f} "
              f"elapsed={time.time()-t0:.1f}s", flush=True)

    checkpoint(0)
    for step in range(1, steps + 1):
        x = prepend_crossfade_batched(delta, car, cf).clamp(-1, 1)
        if lam > 0:
            # SPEC-CHANGE-012 evasion-correct objective, per-sample (each trigger's
            # CE:detector ratio == the single-run objective): minimise
            #   mean_b[ CE_b - lambda*log(1 - p_det_b) ]  ->  p_det drops below tau_op.
            ce_vec = teacher_forced_ce_batched(model, meta, x, tgt_texts, mel_cfg=mel_cfg, per_sample=True)
            p_det = detector_prob_grad(detector, x, mel_cfg=ast_cfg)   # (B,)
            loss = ce_vec.mean() + lam * evasion_penalty(p_det).mean()
        else:
            loss = teacher_forced_ce_batched(model, meta, x, tgt_texts, mel_cfg=mel_cfg)
        (grad,) = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta.add_(-alpha * grad.sign())
            delta.clamp_(-eps, eps)
        if step % ckpt_every == 0 or step == steps:
            checkpoint(step)

    if log_fh:
        log_fh.close()
    wall = time.time() - t0
    summaries = []
    for b in range(B):
        summ = {
            "run_id": run_ids[b], "tool": targets[b].tool, "language": carrier_metas[b].get("language"),
            "carrier_uid": carrier_metas[b].get("uid"), "seed": seed, "lambda": lam,
            "steps": steps, "alpha": alpha, "eps": eps, "crossfade_ms": crossfade_ms,
            "delta_len": delta_len, "carrier_seconds": carrier_seconds,
            "wall_s": round(wall, 1), "sec_per_step": round(wall / max(1, steps), 4), "batch": B,
        }
        with torch.no_grad():
            xb = prepend_crossfade_batched(delta[b:b+1], car[b:b+1], cf).clamp(-1, 1)[0]
            ce_b = teacher_forced_ce(model, meta, xb, tgt_texts[b], mel_cfg=mel_cfg).item()
            pdet_b = (float(detector_prob_grad(detector, xb, mel_cfg=ast_cfg).flatten()[0].item())
                      if detector is not None else None)
        summ["final"] = {"step": steps, "ce_loss": ce_b, "target_prob": math.exp(-ce_b),
                         "delta_linf": delta[b].abs().max().item(),
                         **({"p_det": pdet_b} if pdet_b is not None else {}),
                         **({f"act_{k}": v for k, v in measured_actuation(model, meta, xb, targets[b]).items()} if measure_actuation else {})}
        if backup:
            summ["backup"] = _write_backup(delta[b].detach(), car[b], cf, targets[b], carrier_metas[b], summ)
        summaries.append(summ)
    return summaries


# ------------------------------------------------------------------ SC-009 private backup
def _write_backup(delta: torch.Tensor, carrier: torch.Tensor, cf: int, target: Target,
                  carrier_meta: dict, summary: dict) -> dict:
    """Write waveform + payload to melguard-backup/{triggers,payloads} + manifest.
    NEVER writes to git; the caller decides whether to rclone-push to R2."""
    run_id = summary["run_id"]
    trig_dir = BACKUP_ROOT / "triggers"
    pay_dir = BACKUP_ROOT / "payloads"
    trig_dir.mkdir(parents=True, exist_ok=True)
    pay_dir.mkdir(parents=True, exist_ok=True)

    x = prepend_crossfade(delta.to(torch.float32), carrier.to(delta.device, torch.float32), cf).clamp(-1, 1)
    wav_path = trig_dir / f"{run_id}.flac"
    delta_path = trig_dir / f"{run_id}.delta.npy"
    sf.write(wav_path, x.detach().cpu().numpy().astype(np.float32), SR, format="FLAC")
    np.save(delta_path, delta.detach().cpu().numpy().astype(np.float32))

    payload = {
        "run_id": run_id, "tool": target.tool, "target_text": target.target_text,
        "arg_match": target.arg_match, "carrier_meta": carrier_meta,
        "params": {k: summary[k] for k in ("seed", "lambda", "steps", "alpha", "eps",
                                           "crossfade_ms", "delta_len")},
        "final_metrics": summary["final"], "wav": wav_path.name, "delta": delta_path.name,
    }
    pay_path = pay_dir / f"{run_id}.json"
    pay_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    manifest = BACKUP_ROOT / "MANIFEST.sha256"
    with open(manifest, "a", encoding="utf-8") as fh:
        for p in (wav_path, delta_path, pay_path):
            fh.write(f"{_sha256_file(p)}  {p.relative_to(BACKUP_ROOT)}\n")
    return {"wav": str(wav_path), "delta": str(delta_path), "payload": str(pay_path),
            "manifest": str(manifest)}


def r2_push():
    """rclone-copy the private backup to R2 (SC-009). Requires RCLONE_R2_* in env."""
    import subprocess
    conf = {
        "RCLONE_CONFIG_R2_TYPE": "s3", "RCLONE_CONFIG_R2_PROVIDER": "Cloudflare",
        "RCLONE_CONFIG_R2_ACCESS_KEY_ID": os.environ["RCLONE_R2_ACCESS_KEY_ID"],
        "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": os.environ["RCLONE_R2_SECRET_ACCESS_KEY"],
        "RCLONE_CONFIG_R2_ENDPOINT": os.environ["RCLONE_R2_ENDPOINT"],
        "RCLONE_CONFIG_R2_NO_CHECK_BUCKET": "true",
    }
    env = {**os.environ, **conf}
    cmd = ["rclone", "copy", str(BACKUP_ROOT), "r2:melguard-rq3/melguard-backup",
           "--transfers", "8", "--s3-no-check-bucket"]
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


# ------------------------------------------------------------------ carrier for the smoke
def _smoke_carrier() -> tuple[torch.Tensor, dict]:
    """A single on-disk CommonVoice benign clip for the smoke ONLY (throwaway; the
    scientific sweep uses the reproducible held-out pool from attack/carriers.py)."""
    hits = sorted(glob.glob("/workspace/data/clips/commonvoice18/**/*.flac", recursive=True))
    if not hits:
        raise FileNotFoundError("no commonvoice18 clip on disk for the smoke carrier")
    wav, sr = sf.read(hits[0])
    if wav.ndim == 2:
        wav = wav[:, 0]
    assert sr == SR
    # keep total audio well under 30 s
    wav = wav[: 6 * SR]
    return torch.tensor(wav.astype(np.float32)), {"uid": Path(hits[0]).stem, "language": "smoke", "path": hits[0]}


def _smoke_carriers_multi(n: int):
    """n distinct on-disk CommonVoice clips for batch validation ONLY (throwaway)."""
    hits = sorted(glob.glob("/workspace/data/clips/commonvoice18/**/*.flac", recursive=True))[:n]
    carriers, metas = [], []
    for h in hits:
        wav, sr = sf.read(h)
        if wav.ndim == 2:
            wav = wav[:, 0]
        assert sr == SR
        carriers.append(torch.tensor(wav.astype(np.float32)))
        metas.append({"uid": Path(h).stem, "language": "smoke", "path": h})
    return carriers, metas


# ------------------------------------------------------------------ cube + sweep driver
def build_cube(carrier_records: list) -> list:
    """(carrier_record, tool) for all 75 carriers: within each language, sort by
    uid and assign tool = TOOLS_ORDER[i%3] (operator-ratified rotation)."""
    from collections import defaultdict
    by_lang = defaultdict(list)
    for r in sorted(carrier_records, key=lambda r: (r["language"], r["uid"])):
        by_lang[r["language"]].append(r)
    assigns = []
    for lang in sorted(by_lang):
        for i, r in enumerate(by_lang[lang]):
            assigns.append((r, TOOLS_ORDER[i % 3]))
    return assigns


def _load_carrier_wav(rec: dict) -> torch.Tensor:
    from data.config import DATA_ROOT
    wav, sr = sf.read(DATA_ROOT / rec["path"])
    if wav.ndim == 2:
        wav = wav[:, 0]
    assert sr == SR, f"{rec['path']} sr={sr}"
    return torch.tensor(wav.astype(np.float32))


def _done_path(log_jsonl: Path) -> Path:
    return log_jsonl.with_suffix(".done")


def _load_done(log_jsonl: Path) -> set:
    dp = _done_path(log_jsonl)
    if not dp.exists():
        return set()
    return {l.strip() for l in dp.read_text(encoding="utf-8").splitlines() if l.strip()}


def run_sweep(model, meta, assigns: list, seeds: list, *, batch: int = 3, steps: int = 5000,
              ckpt_every: int = 500, eps: float = 0.04, alpha: float = 0.002,
              alpha_final: float | None = None, carrier_seconds: float = 9.0, lam: float = 0.0,
              detector=None, ast_cfg=None, disclosure_ack: bool = False,
              log_jsonl: Path | None = None, tag: str = "phase5_lam0", r2: bool = False,
              resume: bool = True) -> list:
    """Run the (batched) attack over all (seed × carrier) cells. Resumable: batches
    whose run_ids are already in <log>.done are skipped, so a pod restart continues."""
    tb = number_lock()["trigger_budget"]
    done = _load_done(log_jsonl) if (resume and log_jsonl) else set()
    n_batches = (len(assigns) + batch - 1) // batch
    all_summaries = []
    for seed in seeds:
        for bi, start in enumerate(range(0, len(assigns), batch)):
            group = assigns[start:start + batch]
            run_ids = [f"{tag}_{t}_{r['language']}_s{seed}_{r['uid']}" for r, t in group]
            if resume and all(rid in done for rid in run_ids):
                print(f"[sweep] seed={seed} batch {bi+1}/{n_batches} already done; skip", flush=True)
                continue
            carriers = [_load_carrier_wav(r) for r, _ in group]
            metas = [r for r, _ in group]
            targets = [TARGETS[t] for _, t in group]
            summs = run_attack_batched(
                model, meta, carriers, metas, targets, seed=seed, lam=lam, steps=steps,
                ckpt_every=ckpt_every, eps=eps, delta_len=tb["n_samples"],
                crossfade_ms=tb["crossfade_ms"], alpha=alpha, alpha_final=alpha_final,
                carrier_seconds=carrier_seconds, detector=detector, ast_cfg=ast_cfg,
                disclosure_ack=disclosure_ack, log_jsonl=log_jsonl, tag=tag, backup=True,
            )
            all_summaries.extend(summs)
            if log_jsonl:
                with open(_done_path(log_jsonl), "a", encoding="utf-8") as fh:
                    for rid in run_ids:
                        fh.write(rid + "\n")
            if r2:
                r2_push()
            acts = [s["final"].get("act_l_actuate", 0) for s in summs]
            print(f"[sweep] seed={seed} batch {bi+1}/{n_batches} done; act={acts} "
                  f"cum={len(all_summaries)}/{len(assigns)*len(seeds)}", flush=True)
    return all_summaries


# ------------------------------------------------------------------ CLI
def main() -> int:
    nl = number_lock()
    tb, pg = nl["trigger_budget"], nl["pgd"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 carrier, few steps, sanity only")
    ap.add_argument("--tool", default="transfer_money", choices=list(TARGETS))
    ap.add_argument("--seed", type=int, default=nl["attack_seeds"][0])
    ap.add_argument("--steps", type=int, default=pg["steps"])
    ap.add_argument("--ckpt-every", type=int, default=pg["checkpoint_every"])
    ap.add_argument("--alpha", type=float, default=0.002)
    ap.add_argument("--eps", type=float, default=tb["linf_eps"])
    ap.add_argument("--r2", action="store_true", help="rclone-push the backup to R2")
    ap.add_argument("--batch-validate", type=int, default=0, help="validate batched PGD with N carriers (throwaway)")
    ap.add_argument("--carrier-seconds", type=float, default=9.0)
    ap.add_argument("--val-steps", type=int, default=100)
    ap.add_argument("--no-measure", action="store_true", help="skip greedy actuation measurement (clean timing)")
    ap.add_argument("--pilot", action="store_true", help="3-carrier (3-tool) pilot to full steps")
    ap.add_argument("--sweep", action="store_true", help="full lambda=0 sweep (3 seeds x 75 carriers)")
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--alpha-final", type=float, default=None, help="cosine-anneal alpha to this (else constant)")
    ap.add_argument("--run-name", default="pilot", help="pilot log/tag suffix so re-runs don't collide")
    ap.add_argument("--lam", "--lambda", type=float, default=0.0, dest="lam",
                    help="detector-evasion weight lambda_det (Phase 6). >0 needs --start-disclosure-clock")
    ap.add_argument("--detector-weights", default="out/phase4/detector_gd_best.pt",
                    help="frozen Phase-4 g_D weights for the lambda>0 evasion term")
    ap.add_argument("--start-disclosure-clock", action="store_true",
                    help="operator ack: lambda>0 generates an adaptive trigger and STARTS the 90-day clock")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated attack seeds for --sweep (default: all 3; triage = one seed, e.g. 271828)")
    ap.add_argument("--out", default="out/phase5/pgd_smoke.jsonl")
    args = ap.parse_args()

    seed = set_determinism(args.seed)

    # Phase-6 (lambda>0): load the FROZEN detector g_D. Hard-gated on the operator
    # ack because generating a lambda>0 trigger starts the 90-day disclosure clock.
    detector, ast_cfg = (None, None)
    if args.lam > 0:
        _assert_disclosure_ack(args.lam, args.start_disclosure_clock)
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[pgd] lambda={args.lam:g} (Phase 6): loading FROZEN g_D from {args.detector_weights}")
        detector, ast_cfg = load_attack_detector(args.detector_weights, device=dev)

    if args.pilot or args.sweep:
        from attack.carriers import load_carriers
        assigns = build_cube(load_carriers())
        which = "pilot" if args.pilot else "sweep"
        lam, pdir = args.lam, ("out/phase5" if args.lam == 0 else "out/phase6")
        print(f"[pgd] {which}: loading SALMONN (f_S); lambda={lam:g} alpha={args.alpha} "
              f"steps={args.steps} carrier_s={args.carrier_seconds} batch={args.batch}")
        model, meta = load_salmonn(verify=False)
        if args.pilot:
            subset = [(r, t) for (r, t) in assigns if r["language"] == "en"][:3]  # spans all 3 tools
            run_sweep(model, meta, subset, [args.seed], batch=3, steps=args.steps,
                      ckpt_every=args.ckpt_every, eps=args.eps, alpha=args.alpha,
                      alpha_final=args.alpha_final, carrier_seconds=args.carrier_seconds, lam=lam,
                      detector=detector, ast_cfg=ast_cfg, disclosure_ack=args.start_disclosure_clock,
                      log_jsonl=Path(f"{pdir}/pgd_{args.run_name}.jsonl"),
                      tag=f"{_lam_tag(lam)}_{args.run_name}", r2=args.r2)
        else:
            log = (Path("out/phase5/pgd_sweep_lam0.jsonl") if lam == 0
                   else Path(f"out/phase6/pgd_sweep_lam{lam:g}.jsonl"))
            sweep_seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
                           else number_lock()["attack_seeds"])   # triage: one seed (SC-013)
            run_sweep(model, meta, assigns, sweep_seeds, batch=args.batch,
                      steps=args.steps, ckpt_every=args.ckpt_every, eps=args.eps, alpha=args.alpha,
                      alpha_final=args.alpha_final, carrier_seconds=args.carrier_seconds, lam=lam,
                      detector=detector, ast_cfg=ast_cfg, disclosure_ack=args.start_disclosure_clock,
                      log_jsonl=log, tag=_lam_tag(lam), r2=args.r2)
        return 0

    if args.batch_validate:
        B = args.batch_validate
        tools = list(TARGETS)
        print(f"[pgd] BATCH VALIDATE B={B} carrier_s={args.carrier_seconds} seed={seed}; loading SALMONN (f_S) ...")
        model, meta = load_salmonn(verify=False)
        carriers, metas = _smoke_carriers_multi(B)
        targets = [TARGETS[tools[i % 3]] for i in range(B)]  # i%3 rotation
        torch.cuda.reset_peak_memory_stats()
        out_path = Path("out/phase5/pgd_batch_validate.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summs = run_attack_batched(
            model, meta, carriers, metas, targets, seed=seed, lam=0.0, steps=args.val_steps,
            ckpt_every=max(1, args.val_steps // 2), eps=args.eps,
            delta_len=number_lock()["trigger_budget"]["n_samples"],
            crossfade_ms=number_lock()["trigger_budget"]["crossfade_ms"], alpha=0.006,
            carrier_seconds=args.carrier_seconds, log_jsonl=out_path,
            tag="phase5_lam0_batchval", backup=False, measure_actuation=not args.no_measure,
        )
        peak = torch.cuda.max_memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(json.dumps({
            "batch": B, "peak_reserved_gib": round(peak, 2), "total_gib": round(total, 1),
            "fits": peak < total, "sec_per_step": summs[0]["sec_per_step"],
            "per_sample": [{"tool": s["tool"], "final_ce": s["final"]["ce_loss"],
                            "target_prob": round(s["final"]["target_prob"], 3),
                            "linf": round(s["final"]["delta_linf"], 4),
                            "act": s["final"].get("act_l_actuate")} for s in summs],
        }, indent=2))
        return 0
    if args.smoke:
        steps = args.steps if args.steps != pg["steps"] else 50
        ckpt = args.ckpt_every if args.ckpt_every != pg["checkpoint_every"] else 10
        alpha = args.alpha if args.alpha != 0.002 else 0.008
        tag = f"{_lam_tag(args.lam)}_smoke"
    else:
        steps, ckpt, alpha, tag = args.steps, args.ckpt_every, args.alpha, _lam_tag(args.lam)

    print(f"[pgd] loading SALMONN (f_S); tool={args.tool} seed={seed} steps={steps} "
          f"alpha={alpha} eps={args.eps} lambda={args.lam:g} smoke={args.smoke}")
    model, meta = load_salmonn(verify=False)

    carrier, cmeta = _smoke_carrier()  # smoke path; sweep will pass pool carriers
    target = TARGETS[args.tool]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = run_attack(
        model, meta, carrier, cmeta, target, seed=seed, lam=args.lam, steps=steps,
        ckpt_every=ckpt, eps=args.eps, delta_len=tb["n_samples"],
        crossfade_ms=tb["crossfade_ms"], alpha=alpha, detector=detector, ast_cfg=ast_cfg,
        disclosure_ack=args.start_disclosure_clock, log_jsonl=out_path, tag=tag,
    )
    # summary artifact (metrics only; git-safe)
    summ_path = out_path.with_suffix(".summary.json")
    summ_path.write_text(json.dumps({k: v for k, v in summary.items() if k != "checkpoints"}, indent=2),
                         encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("run_id", "wall_s", "sec_per_step", "init", "final", "backup")}, indent=2))

    if args.r2:
        res = r2_push()
        print(f"[pgd] r2 push rc={res.returncode} {res.stderr[-200:] if res.stderr else ''}")
    return 0


if __name__ == "__main__":
    rc = main()
    os._exit(rc)
