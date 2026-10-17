"""Differentiable audio front-end for the Phase 5-6 joint-PGD attack.

The deployed target f_S (``salmonn/model.py``) featurises the waveform with a
numpy ``WhisperFeatureExtractor`` and bridges BEATs through ``torch.from_numpy``,
and its ``encode_audio`` is wrapped in ``@torch.inference_mode()``. All three
sever the autograd path from the LLM loss back to the waveform perturbation
``delta`` that the white-box attacker optimises (spec.lock.md §5, threat model).

This module rebuilds the **same** front-end so gradients reach the waveform,
changing no pinned weight or hyper-parameter:

* ``torch_whisper_logmel`` reimplements ``WhisperFeatureExtractor`` in pure torch
  (pad/trim to 30 s -> STFT -> power -> pinned mel filterbank -> log10 -> dynamic
  range -> scale). ``n_fft``, ``hop_length``, ``n_samples`` and the mel filter
  matrix are read from the *loaded pinned extractor*, so nothing is hardcoded.

* BEATs already computes its filterbank in torch
  (``torchaudio.compliance.kaldi.fbank`` inside
  ``salmonn/upstream/beats/BEATs.py::preprocess``; ``dither=0.0`` so it is
  deterministic and differentiable). We reuse the model's own
  ``beats.extract_features`` on a live tensor, so the BEATs branch is
  bit-identical to deployment by construction — the only fix it needs is to stop
  going through ``torch.from_numpy`` and ``inference_mode``.

* ``encode_audio_grad`` mirrors ``MelGuardSALMONN.encode_audio`` line-for-line
  but (a) runs under autograd, (b) uses the torch mel + a live wav tensor into
  BEATs, reusing the model's own submodules (``speech_encoder``, ``ln_speech``,
  ``beats``, ``ln_audio``, ``speech_Qformer``, ``speech_llama_proj``) so f_S is
  untouched. Its output feeds the LLM exactly as the deployed path does.

``forward_equivalence`` and ``grad_reaches_waveform`` implement the checks the
phase requires (numerical parity vs the deployed numpy front-end, and a live
gradient to the waveform).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- #
# Whisper log-mel, reimplemented in torch to match the pinned extractor exactly. #
# ----------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WhisperMelConfig:
    n_fft: int
    hop_length: int
    n_samples: int          # pad/trim target (chunk_length * sr, e.g. 480000)
    mel_filters: torch.Tensor  # (n_mels, n_freqs), i.e. HF mel_filters.T


def whisper_mel_config(model, *, device=None, dtype: torch.dtype = torch.float32) -> WhisperMelConfig:
    """Extract the mel-spectrogram config from the model's loaded, pinned
    ``WhisperFeatureExtractor``. Read from the object; never hardcoded."""
    fe = model.feature_extractor
    device = device or model.device_str
    # HF stores mel_filters as (n_freqs, n_mels); we want (n_mels, n_freqs).
    mel = torch.as_tensor(fe.mel_filters, dtype=dtype, device=device).T.contiguous()
    return WhisperMelConfig(
        n_fft=int(fe.n_fft),
        hop_length=int(fe.hop_length),
        n_samples=int(fe.n_samples),
        mel_filters=mel,
    )


def torch_whisper_logmel(
    wav: torch.Tensor,
    cfg: WhisperMelConfig,
    *,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Differentiable equivalent of ``WhisperFeatureExtractor(wav).input_features``.

    ``wav``: (T,) or (B, T) float waveform @16 kHz on the mel-config device, may
    carry grad. Returns (B, n_mels, nb_max_frames) — the log-mel the Whisper
    encoder consumes. Computed in fp32 for fidelity, then cast to ``out_dtype``.
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    work_dtype = torch.float32
    wav = wav.to(work_dtype)

    # Pad/trim to 30 s exactly as WhisperFeatureExtractor (padding="max_length",
    # truncation=True, padding_value=0.0).
    n = cfg.n_samples
    T = wav.shape[-1]
    if T < n:
        wav = F.pad(wav, (0, n - T))
    elif T > n:
        wav = wav[..., :n]

    window = torch.hann_window(cfg.n_fft, periodic=True, device=wav.device, dtype=work_dtype)
    stft = torch.stft(
        wav, cfg.n_fft, hop_length=cfg.hop_length, window=window, return_complex=True
    )  # center=True, pad_mode="reflect", onesided=True  -> (B, n_freqs, frames+1)
    magnitudes = stft[..., :-1].abs() ** 2  # drop last frame; power spectrum

    mel = cfg.mel_filters.to(work_dtype)          # (n_mels, n_freqs)
    mel_spec = torch.matmul(mel, magnitudes)      # (B, n_mels, frames)

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    # Dynamic-range compression: floor at (per-clip max - 8). B==1 in the attack,
    # so per-sample amax matches the deployed single-clip global max.
    log_spec = torch.maximum(log_spec, log_spec.amax(dim=(-2, -1), keepdim=True) - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.to(out_dtype)


# ----------------------------------------------------------------------------- #
# Differentiable encode_audio (mirror of MelGuardSALMONN.encode_audio).          #
# ----------------------------------------------------------------------------- #
def encode_audio_grad(
    model,
    wav: torch.Tensor,
    *,
    mel_cfg: WhisperMelConfig | None = None,
) -> torch.Tensor:
    """Autograd-enabled ``encode_audio``: waveform (B, T) or (T,) -> speech embeds
    in the LLM space (B, L, llm_hidden). Identical maths to the deployed path but
    differentiable w.r.t. ``wav``. ``model`` is a built ``MelGuardSALMONN``."""
    device = model.device_str
    dt = model.encoder_dtype  # fp16
    if mel_cfg is None:
        mel_cfg = whisper_mel_config(model, device=device)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav = wav.to(device)

    # --- Whisper branch: torch log-mel -> fp16 speech encoder ---
    spectrogram = torch_whisper_logmel(wav, mel_cfg, out_dtype=dt)
    speech_embeds = model.speech_encoder(spectrogram, return_dict=True).last_hidden_state

    # --- BEATs branch: live fp32 tensor through the model's own kaldi fbank ---
    raw_wav = wav.to(torch.float32)
    audio_padding_mask = torch.zeros_like(raw_wav, dtype=torch.bool)
    audio_embeds, _ = model.beats.extract_features(
        raw_wav, padding_mask=audio_padding_mask, feature_only=True
    )
    audio_embeds = audio_embeds.to(dt)

    # --- everything below is verbatim from encode_audio ---
    speech_embeds = model.ln_speech(speech_embeds)
    audio_embeds = model.ln_audio(audio_embeds)
    audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
    speech_embeds = torch.cat([speech_embeds, audio_embeds], dim=-1)

    B, T, C = speech_embeds.shape
    kernel = round(T * model.second_per_frame / 30.0)
    stride = round(T * model.second_stride / 30.0)
    kernel = (1, kernel)
    stride = (1, stride)
    speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
    speech_embeds_overlap = F.unfold(
        speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride
    )
    _, _, Ln = speech_embeds_overlap.shape
    speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], Ln)
    speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
    speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
    speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

    query_tokens = model.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
    query_output = model.speech_Qformer.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=speech_embeds,
        encoder_attention_mask=speech_atts,
        return_dict=True,
    )
    speech_embeds = model.speech_llama_proj(query_output.last_hidden_state)
    speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()
    return speech_embeds


# ----------------------------------------------------------------------------- #
# Verification helpers (the phase's forward-equivalence + gradient checks).       #
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def _cosine_per_token(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.reshape(-1, a.shape[-1]).float()
    b = b.reshape(-1, b.shape[-1]).float()
    return F.cosine_similarity(a, b, dim=-1)


def forward_equivalence(model, wav_np) -> dict:
    """Compare the torch front-end against the deployed numpy front-end on one
    clip. Returns a metrics dict (mel diff, embed diff, cosine, dtype checks)."""
    import numpy as np

    device = model.device_str
    wav_np = np.asarray(wav_np, dtype=np.float32)

    # Reference: deployed numpy path (inference_mode inside).
    ref_mel = model.feature_extractor(
        wav_np, return_tensors="pt", sampling_rate=16000
    ).input_features.to(device=device, dtype=torch.float32)
    ref_embeds = model.encode_audio(wav_np, device=device).float()

    # Torch path.
    wav_t = torch.tensor(wav_np, device=device, dtype=torch.float32)
    mel_cfg = whisper_mel_config(model, device=device)
    with torch.no_grad():
        torch_mel = torch_whisper_logmel(wav_t, mel_cfg, out_dtype=torch.float32)
        torch_embeds = encode_audio_grad(model, wav_t, mel_cfg=mel_cfg).float()

    mel_abs = (torch_mel - ref_mel).abs()
    emb_abs = (torch_embeds - ref_embeds).abs()
    ref_scale = ref_embeds.abs().mean().clamp_min(1e-6)
    cos = _cosine_per_token(torch_embeds, ref_embeds)

    return {
        "mel_shape": list(torch_mel.shape),
        "mel_max_abs_diff": mel_abs.max().item(),
        "mel_mean_abs_diff": mel_abs.mean().item(),
        "embed_shape": list(torch_embeds.shape),
        "embed_max_abs_diff": emb_abs.max().item(),
        "embed_mean_abs_diff": emb_abs.mean().item(),
        "embed_rel_mean_diff": (emb_abs.mean() / ref_scale).item(),
        "embed_cosine_min": cos.min().item(),
        "embed_cosine_mean": cos.mean().item(),
    }


def grad_reaches_waveform(model, wav_np, *, target_len: int = 16000) -> dict:
    """Confirm a live gradient flows to the waveform through *both* encoder
    branches. Splits the input into a leaf ``delta`` (first ``target_len``
    samples, as the attack does) + a frozen carrier tail, and backprops a scalar
    of the speech embeddings."""
    import numpy as np

    device = model.device_str
    wav_np = np.asarray(wav_np, dtype=np.float32)
    wav_t = torch.tensor(wav_np, device=device, dtype=torch.float32)
    n = min(target_len, wav_t.numel())

    delta = wav_t[:n].clone().detach().requires_grad_(True)
    carrier = wav_t[n:].clone().detach()
    full = torch.cat([delta, carrier], dim=0)

    emb = encode_audio_grad(model, full)
    loss = emb.float().pow(2).mean()
    loss.backward()

    g = delta.grad
    return {
        "loss": loss.item(),
        "delta_len": int(n),
        "grad_is_finite": bool(g is not None and torch.isfinite(g).all().item()),
        "grad_abs_sum": float(g.abs().sum().item()) if g is not None else 0.0,
        "grad_abs_max": float(g.abs().max().item()) if g is not None else 0.0,
        "grad_nonzero_frac": float((g != 0).float().mean().item()) if g is not None else 0.0,
    }
