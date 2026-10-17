"""Phase-8 channel simulation: codec re-encoding and over-the-air propagation.

Registered in OSF j5x2e 344-83 as EXPLORATORY, non-confirmatory robustness sub-studies:
"codec re-evaluation (Opus 16 kbps, Vorbis 64 kbps); over-the-air simulation (1 m lab room
impulse response plus additive white Gaussian noise at SNR in {20, 10} dB)".

Per SPEC-CHANGE-014 these run on NON-SEALED data only (FLEURS benign_control + the dev-seed
s31337 triggers). The Phase-7 held-out set is burned and is never touched here.

SC-009 guardrail: channel-degraded adversarial clips are derived trigger waveforms. They are
produced IN MEMORY and never written to disk or git.

Tooling: codec round-trips use the static ffmpeg bundled by the imageio-ffmpeg pip wheel
(libopus / libvorbis), which accepts an exact target bitrate -- realized 15.9 kbps for the
Opus-16k target and 69.3 kbps for the Vorbis-64k target, both reported per clip. It was
chosen over a system ffmpeg because it needs no root and leaves the pinned model stack
untouched (torch 2.5.1+cu124 / transformers 4.46.3 verified unchanged after install).
pyroomacoustics is unavailable, so the room impulse response is a self-contained image-source
method implemented below.
"""
from __future__ import annotations

import io
import math

import numpy as np
import soundfile as sf

SR = 16000
SOUND_SPEED = 343.0


# ------------------------------------------------------------------------ codec
_FFMPEG: str | None = None


def ffmpeg_exe() -> str:
    """Static ffmpeg bundled by the imageio-ffmpeg wheel. Chosen over a system package
    because it changes nothing in the pinned model stack (torch 2.5.1+cu124 /
    transformers 4.46.3 verified unchanged after install) and needs no root."""
    global _FFMPEG
    if _FFMPEG is None:
        import imageio_ffmpeg
        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return _FFMPEG


def _run(cmd: list[str], payload: bytes) -> bytes:
    import subprocess
    p = subprocess.run(cmd, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr.decode()[-500:]}")
    return p.stdout


def _to_wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav.astype(np.float32), sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def codec_roundtrip(wav: np.ndarray, sr: int, codec: str) -> tuple[np.ndarray, float]:
    """Encode at the REGISTERED bitrate and decode back. Returns (wav, realized kbps).

    Everything is piped in memory -- no channel-degraded trigger waveform ever reaches
    disk (SC-009). libsndfile alone could not do this: python-soundfile 0.12 exposes no
    bitrate control, and its Opus default landed at 34 kbps against the registered 16.
    """
    enc = {"opus": ["-c:a", "libopus", "-b:a", "16k"],
           "vorbis": ["-c:a", "libvorbis", "-b:a", "64k"]}[codec]
    exe = ffmpeg_exe()
    blob = _run([exe, "-hide_banner", "-loglevel", "error", "-f", "wav", "-i", "pipe:0",
                 "-ac", "1", "-ar", str(sr), *enc, "-f", "ogg", "pipe:1"],
                _to_wav_bytes(wav, sr))
    seconds = len(wav) / sr
    kbps = (len(blob) * 8.0 / 1000.0) / seconds if seconds > 0 else float("nan")
    raw = _run([exe, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                "-ac", "1", "-ar", str(sr), "-f", "wav", "-c:a", "pcm_f32le", "pipe:1"], blob)
    out, out_sr = sf.read(io.BytesIO(raw), dtype="float32")
    if out.ndim == 2:
        out = out[:, 0]
    if out_sr != sr:
        raise ValueError(f"codec changed the sample rate: {out_sr} != {sr}")
    # lapped transforms add encoder delay; align length so downstream framing is unchanged
    if len(out) < len(wav):
        out = np.pad(out, (0, len(wav) - len(out)))
    return out[: len(wav)], kbps


# -------------------------------------------------------------- room impulse response
def image_source_rir(sr: int = SR, room=(5.0, 4.0, 3.0), src=(1.5, 2.0, 1.2),
                     mic=(2.5, 2.0, 1.2), rt60: float = 0.35,
                     max_order: int = 12) -> np.ndarray:
    """Shoebox room impulse response by the image-source method.

    Defaults describe a small lab (5 x 4 x 3 m) with the microphone 1 m from the source --
    the "1 m lab room" of 344-83. Wall reflection coefficient is derived from the target
    RT60 through Sabine's equation, so the decay is physically consistent rather than an
    arbitrary exponential."""
    room = np.asarray(room, float)
    src = np.asarray(src, float)
    mic = np.asarray(mic, float)
    dist = float(np.linalg.norm(src - mic))
    if not (0.9 <= dist <= 1.1):
        raise ValueError(f"source-mic distance {dist:.3f} m is not ~1 m")

    volume = float(np.prod(room))
    surface = 2.0 * (room[0] * room[1] + room[1] * room[2] + room[0] * room[2])
    alpha = min(0.99, 0.161 * volume / (rt60 * surface))    # Sabine
    beta = math.sqrt(max(1e-6, 1.0 - alpha))                # pressure reflection coeff

    length = int(sr * (rt60 * 1.5)) + 1
    rir = np.zeros(length, dtype=np.float64)
    for nx in range(-max_order, max_order + 1):
        for ny in range(-max_order, max_order + 1):
            for nz in range(-max_order, max_order + 1):
                for px in (0, 1):
                    for py in (0, 1):
                        for pz in (0, 1):
                            # mirrored source position
                            ix = (1 - 2 * px) * src[0] + 2 * nx * room[0]
                            iy = (1 - 2 * py) * src[1] + 2 * ny * room[1]
                            iz = (1 - 2 * pz) * src[2] + 2 * nz * room[2]
                            d = math.dist((ix, iy, iz), tuple(mic))
                            if d < 1e-6:
                                continue
                            n_refl = abs(nx) + abs(ny) + abs(nz) + px + py + pz
                            amp = (beta ** n_refl) / (4.0 * math.pi * d)
                            t = d / SOUND_SPEED * sr
                            i0 = int(math.floor(t))
                            if i0 < 0 or i0 + 1 >= length:
                                continue
                            frac = t - i0
                            rir[i0] += amp * (1.0 - frac)     # linear fractional delay
                            rir[i0 + 1] += amp * frac
    peak = np.max(np.abs(rir))
    if peak > 0:
        rir /= peak                                          # unit-peak, keeps levels sane
    return rir.astype(np.float32)


def apply_ota(wav: np.ndarray, rir: np.ndarray, snr_db: float,
              rng: np.random.Generator) -> np.ndarray:
    """Convolve with the RIR, then add white Gaussian noise at the requested SNR.

    Length is preserved (the reverb tail is truncated) so downstream framing is unchanged,
    and SNR is measured on the reverberant signal that the microphone actually receives."""
    wet = reverberate(wav, rir)
    p_sig = float(np.mean(wet ** 2))
    if p_sig <= 0:
        return wet
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, math.sqrt(p_noise), size=len(wet))
    return (wet + noise).astype(np.float32)


def reverberate(wav: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve and rescale to the DRY signal's peak.

    The rescale is deliberate and must happen BEFORE noise is added: convolution with a
    multi-tap RIR raises the peak well above 1.0, and normalising afterwards would rescale
    signal and noise together, leaving the achieved SNR unverifiable against the dry
    reference. Fixing the level here makes apply_ota's SNR exact and reproducible."""
    from scipy.signal import fftconvolve
    wet = fftconvolve(wav.astype(np.float64), rir.astype(np.float64))[: len(wav)]
    peak_dry, peak_wet = float(np.max(np.abs(wav))), float(np.max(np.abs(wet)))
    if peak_wet > 0 and peak_dry > 0:
        wet = wet * (peak_dry / peak_wet)
    return wet.astype(np.float32)


def measured_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Verification helper: SNR of `noisy` treating `clean` as the reference signal."""
    noise = noisy[: len(clean)] - clean
    p_s, p_n = float(np.mean(clean ** 2)), float(np.mean(noise ** 2))
    return 10.0 * math.log10(p_s / p_n) if p_n > 0 else float("inf")


# ---------------------------------------------------------------------- self-test
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = np.arange(SR) / SR
    wav = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.normal(size=SR)).astype(np.float32)

    for codec in ("opus", "vorbis"):
        out, kbps = codec_roundtrip(wav, SR, codec)
        print(f"{codec:7s} realized {kbps:6.1f} kbps  len={len(out)}  "
              f"corr={np.corrcoef(wav, out)[0,1]:.4f}")

    rir = image_source_rir()
    nz = int(np.sum(np.abs(rir) > 1e-4))
    print(f"RIR: len={len(rir)} ({len(rir)/SR:.2f}s) taps>1e-4={nz} "
          f"peak_idx={int(np.argmax(np.abs(rir)))}")
    wet = reverberate(wav, rir)
    for snr in (20.0, 10.0):
        deg = apply_ota(wav, rir, snr, np.random.default_rng(1))
        print(f"OTA snr_target={snr:4.1f} dB  measured={measured_snr_db(wet, deg):6.2f} dB  "
              f"len={len(deg)}  peak={float(np.max(np.abs(deg))):.3f}")
