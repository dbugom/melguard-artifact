"""Differentiable detector front-end for the Phase-6 (lambda>0) joint-PGD attack.

The deployed detector g_D (`detector/model.py::MelGuardDetector`) featurises audio
with a numpy `ASTFeatureExtractor` (via `detector/data.py`), which detaches the
graph, so the attacker's detector-evasion term `log(1 - g_D(x))` gets no gradient
to the waveform delta. `MelGuardDetector.forward` itself is already differentiable
(ASTModel encoder + head); only the front-end breaks the path.

This module rebuilds the AST log-mel in torch, changing no pinned weight:

* `torch_ast_logmel` reproduces `ASTFeatureExtractor` — `torchaudio.compliance.
  kaldi.fbank` (hanning window, 128 mel bins, dither=0.0 -> deterministic +
  differentiable, NO 2**15 scaling — the AST extractor comments that line out,
  unlike BEATs), pad/truncate to `max_length`, then AudioSet normalisation
  `(x - mean) / (2*std)`. num_mel_bins / max_length / mean / std are read from the
  loaded pinned extractor, nothing hardcoded.
* `detector_prob_grad` = torch_ast_logmel -> MelGuardDetector.forward -> sigmoid,
  differentiable w.r.t. the waveform, so `-lambda * log(1 - g_D(x))` reaches delta.

Mirrors `attack/frontend.py` (the SALMONN front-end) for the detector side.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio.compliance.kaldi as ta_kaldi


@dataclass(frozen=True)
class ASTMelConfig:
    num_mel_bins: int
    max_length: int
    mean: float
    std: float
    do_normalize: bool


def ast_mel_config(feature_extractor) -> ASTMelConfig:
    """Read the mel config from the model's loaded, pinned ASTFeatureExtractor."""
    fe = feature_extractor
    return ASTMelConfig(
        num_mel_bins=int(fe.num_mel_bins),
        max_length=int(fe.max_length),
        mean=float(fe.mean),
        std=float(fe.std),
        do_normalize=bool(fe.do_normalize),
    )


def torch_ast_logmel(wav: torch.Tensor, cfg: ASTMelConfig) -> torch.Tensor:
    """Differentiable equivalent of ``ASTFeatureExtractor(wav).input_values``.

    ``wav``: (T,) or (B, T) float @16 kHz, may carry grad. Returns
    (B, max_length, num_mel_bins). Kaldi fbank is computed per row (it takes a
    single waveform), then padded/truncated and normalised."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    feats = []
    for row in wav:  # ta_kaldi.fbank expects (1, T)
        fbank = ta_kaldi.fbank(
            row.unsqueeze(0),
            sample_frequency=16000,
            window_type="hanning",
            num_mel_bins=cfg.num_mel_bins,
        )  # (n_frames, num_mel_bins)
        n = fbank.shape[0]
        if n < cfg.max_length:
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, cfg.max_length - n))
        elif n > cfg.max_length:
            fbank = fbank[: cfg.max_length]
        feats.append(fbank)
    out = torch.stack(feats, dim=0)  # (B, max_length, num_mel_bins)
    if cfg.do_normalize:
        out = (out - cfg.mean) / (cfg.std * 2)
    return out


def detector_logit_grad(detector, wav: torch.Tensor, *, mel_cfg: ASTMelConfig | None = None) -> torch.Tensor:
    """Autograd-enabled g_D logit(s) for waveform(s). ``detector`` = MelGuardDetector."""
    if mel_cfg is None:
        raise ValueError("pass mel_cfg from ast_mel_config(feature_extractor)")
    input_values = torch_ast_logmel(wav.to(next(detector.parameters()).device), mel_cfg)
    return detector.forward(input_values)  # (B,) raw logits


def detector_prob_grad(detector, wav: torch.Tensor, *, mel_cfg: ASTMelConfig | None = None) -> torch.Tensor:
    """Autograd-enabled g_D probability p_det(x) in [0,1]. This is what the Phase-6
    detector-evasion term log(1 - p_det) differentiates through."""
    return torch.sigmoid(detector_logit_grad(detector, wav, mel_cfg=mel_cfg))


# ----------------------------------------------------------------------------- #
# Verification helpers (forward-equivalence vs numpy ASTFeatureExtractor + grad). #
# ----------------------------------------------------------------------------- #
def forward_equivalence(detector, feature_extractor, wav_np) -> dict:
    """Compare the torch AST front-end + g_D score against the deployed numpy path."""
    import numpy as np

    device = next(detector.parameters()).device
    wav_np = np.asarray(wav_np, dtype=np.float32)
    cfg = ast_mel_config(feature_extractor)

    # Reference: deployed numpy ASTFeatureExtractor + detector
    ref_iv = feature_extractor(wav_np, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        ref_logit = detector.forward(ref_iv)
        ref_prob = torch.sigmoid(ref_logit)

    # Torch path
    wav_t = torch.tensor(wav_np, device=device, dtype=torch.float32)
    with torch.no_grad():
        torch_iv = torch_ast_logmel(wav_t, cfg)
        torch_logit = detector_logit_grad(detector, wav_t, mel_cfg=cfg)
        torch_prob = torch.sigmoid(torch_logit)

    iv_abs = (torch_iv - ref_iv).abs()
    return {
        "iv_shape": list(torch_iv.shape),
        "iv_max_abs_diff": iv_abs.max().item(),
        "iv_mean_abs_diff": iv_abs.mean().item(),
        "ref_logit": ref_logit.flatten().tolist(),
        "torch_logit": torch_logit.flatten().tolist(),
        "ref_prob": ref_prob.flatten().tolist(),
        "torch_prob": torch_prob.flatten().tolist(),
        "prob_abs_diff": (torch_prob - ref_prob).abs().max().item(),
    }


def grad_reaches_waveform(detector, wav_np, feature_extractor, *, target_len: int = 16000) -> dict:
    """Confirm a live gradient flows from g_D's score to a delta at the head of the
    waveform (the Phase-6 evasion term's gradient path)."""
    import numpy as np

    device = next(detector.parameters()).device
    wav_np = np.asarray(wav_np, dtype=np.float32)
    cfg = ast_mel_config(feature_extractor)
    wav_t = torch.tensor(wav_np, device=device, dtype=torch.float32)
    n = min(target_len, wav_t.numel())
    delta = wav_t[:n].clone().detach().requires_grad_(True)
    carrier = wav_t[n:].clone().detach()
    full = torch.cat([delta, carrier], dim=0)

    p = detector_prob_grad(detector, full, mel_cfg=cfg)          # (1,)
    loss = torch.log1p(-p.clamp(max=1 - 1e-6)).sum()             # log(1 - p_det) — the evasion term
    loss.backward()
    g = delta.grad
    return {
        "p_det": p.flatten().tolist(),
        "evasion_term_log1m_p": loss.item(),
        "grad_is_finite": bool(g is not None and torch.isfinite(g).all().item()),
        "grad_abs_sum": float(g.abs().sum().item()) if g is not None else 0.0,
        "grad_nonzero_frac": float((g != 0).float().mean().item()) if g is not None else 0.0,
    }
