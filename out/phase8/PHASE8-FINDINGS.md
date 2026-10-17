# Phase 8 — robustness sub-studies (EXPLORATORY, non-confirmatory, 344-83)

All four registered robustness sub-studies, run on **NON-SEALED** data per SPEC-CHANGE-014:
benign negatives = FLEURS `benign_control` (900; 344-47 designates FLEURS "used only in
robustness sub-studies"), adversarial positives = the dev-seed **s31337** triggers (λ ∈ {0, 1};
λ=0.1/10 were triage, held-seed only). **The Phase-7 sealed held-out set is burned and is never
touched here** — a guard in the harness asserts no sealed role is loaded. These results are
exploratory and do not enter the Phase-7 headline.

Frozen Phase-7 detector (v1 hardened, τ_op = 0.00032981). Code: `robustness/channels.py`,
`robustness/run_channel_eval.py`, `robustness/qwen_transfer.py`. Artifacts:
`out/phase8/channel_*.json`, `out/phase8/act/channel_*.json`, `out/phase8/qwen2audio_transfer.json`.

---

## Headline

Two independent channels each break the attack, but for opposite reasons, and only one leaves
the detector usable:

- **Lossy codec transcoding is incidentally protective.** Opus-16k / Vorbis-64k re-encoding
  destroys the sub-second adversarial perturbation: trigger actuation on SALMONN collapses from
  ~80% to **0%**, detector AUC on the triggers *rises* toward 1.0, and benign FPR is unchanged.
- **Over-the-air propagation breaks everything, including the defender.** A 1 m room impulse
  response + AWGN destroys the attack too (actuation → 0% at 20 dB; not measured at 10 dB), but it also collapses the detector:
  benign FPR rises to 52% (SNR 20 dB) / 76% (SNR 10 dB) and AUC falls to chance. OTA is **not** a
  usable defense — the clean-trained detector is not OTA-robust.
- **The white-box attack does not transfer to Qwen2-Audio** as a *targeted* attack (1.3%), but
  the triggers still induce **restricted tool-calling 72% of the time**. Without a Qwen clean or
  noise baseline, this 72% is consistent with prompt saturation and is not evidence of transfer
  (corrected 2026-10-17; see §4).

---

## 1. Codec re-evaluation (Opus 16 kbps, Vorbis 64 kbps)

Realized bitrates (ffmpeg libopus/libvorbis, measured per clip): **Opus 15.9 kbps**, **Vorbis
69.3 kbps** — on target. Round-trips are piped in memory; no degraded trigger touches disk (SC-009).

| channel | benign FPR | AUC λ=0 | AUC λ=1 | SALMONN actuation λ=0 / λ=1 |
|---|---|---|---|---|
| clean | 0.0333 (30/900) | 1.0000 | 0.9635 | 0.813 / 0.800 (baseline) |
| Opus 16k | 0.0322 (29/900) | 0.9997 | **0.9992** | **0.000 / 0.000** |
| Vorbis 64k | 0.0278 (25/900) | 1.0000 | **0.9891** | — |

**Interpretation.** The λ=1 adversarial perturbation is a sub-second, ℓ∞ ≤ 0.04 signal that a
lossy perceptual codec quantises away. Two consequences: (a) the detector finds the transcoded
triggers *more* separable (AUC 0.9635 → 0.999), and (b) the attack simply stops working —
SALMONN no longer emits the target tool call, so actuation is 0. In any real audio pipeline that
transcodes (VoIP, streaming, messaging), this specific attack degrades before it reaches the
model. Benign FPR is essentially unchanged, so codec transcoding costs the defender nothing.
This is the one genuinely favourable robustness result for the detector — but note it is a
property of the *channel*, not of the detector, and a codec-aware adversary could optimise
through the transcode.

## 2. Over-the-air simulation (1 m lab RIR + AWGN, SNR ∈ {20, 10} dB)

RIR by a self-contained image-source method (5×4×3 m shoebox, mic 1 m from source, RT60 0.35 s
via Sabine). SNR is exact against the reverberant reference (verified 20.04 / 10.04 dB).

| channel | benign FPR | AUC λ=0 | AUC λ=1 | SALMONN actuation λ=0 / λ=1 |
|---|---|---|---|---|
| clean | 0.0333 | 1.0000 | 0.9635 | 0.813 / 0.800 |
| OTA 20 dB | **0.5156 (464/900)** | **0.6849** | **0.6783** | 0.000 / 0.000 |
| OTA 10 dB | **0.7644 (688/900)** | **0.5191** | **0.5207** | — (not measured) |

**Interpretation.** Reverberation + noise is far outside the clean-speech distribution the
detector was trained on. It reads ordinary reverberant benign speech as adversarial (FPR 52% at
20 dB, 76% at 10 dB) and its discrimination falls to chance (AUC 0.52 at 10 dB). The attack is
also destroyed at 20 dB (actuation 0; not measured at 10 dB), but that is cold comfort: **OTA does not defend, it disables** —
the gateway would reject half of all benign speech in a normal room. This is an honest
deployability limit: a passive Mel-spectrogram detector trained on clean corpora cannot be
fielded against real acoustic capture without OTA-augmented training.

**Per-language note (exploratory).** Under OTA the FLEURS-es benign FPR is anomalously high
(98.7% at 20 dB) versus en/zh (26%/30%). Root cause is a corpus property, not the pipeline:
FLEURS-es clips have ~50× the median RMS of FLEURS-en (0.056 vs 0.0011), and the louder signal
interacts with reverberant smearing to push more energy into the bands the detector keys on. The
headline OTA collapse holds regardless of this per-language spread.

## 3. Cross-language (registered as leave-one-language-out; realized as a per-language breakdown)

> **Corrected 2026-10-17.** The earlier table and text here did not match the committed JSONs
> (e.g. clean en λ1 was given as 0.94; OTA-20 zh-CN as ~0.73). The table below is regenerated from
> `out/phase8/channel_*.json`. The registered leave-one-language-out transfer was realized as a
> per-language breakdown of the same all-language detector; no language was held out of training.

Per-language detector AUC (each language's triggers vs its own benign control) and benign FPR:

| channel | en AUC (λ0/λ1) | zh-CN AUC (λ0/λ1) | es AUC (λ0/λ1) | benign FPR en / zh-CN / es |
|---|---|---|---|---|
| clean | 1.00 / 0.89 | 1.00 / 1.00 | 1.00 / 1.00 | 0.100 / 0.000 / 0.000 |
| Opus 16k | 1.00 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 | 0.097 / 0.000 / 0.000 |
| Vorbis 64k | 1.00 / 0.96 | 1.00 / 1.00 | 1.00 / 1.00 | 0.083 / 0.000 / 0.000 |
| OTA 20 dB | 0.86 / 0.85 | 0.88 / 0.89 | 0.29 / 0.23 | 0.260 / 0.300 / 0.987 |
| OTA 10 dB | 0.78 / 0.78 | 0.78 / 0.79 | 0.02 / 0.02 | 0.393 / 0.900 / 1.000 |

The detector is **not** language-uniform: on clean and codec channels all benign false positives
are English, and English is the only language with a λ=1 AUC below 0.90 on the clean channel.
Over the air the collapse is uneven and worst for Spanish. In FLEURS, language is confounded with
each subset's recording conditions (see the §2 es note), so these are per-language-subset
effects. On the sealed Phase-7 set, by contrast, the English benign-FPR excess is small
(0.027 vs 0.020 and 0.018) and no language's AUC falls below the pre-registered 0.80
per-language floor.

## 4. Black-box model transfer to Qwen2-Audio-7B-Instruct (attack-success only)

Per 344-83, attack-success rate only — not detector AUC, since the target model differs. The
150 dev-seed triggers (white-box-optimised on SALMONN) were fed to Qwen2-Audio-7B-Instruct with
the same instruction prompt; success = the reply parses to the trigger's **targeted** restricted
tool call (the strict Phase-7 `l_actuate` predicate).

**SC-011 resolved:** the pinned 7B weights (`Qwen/Qwen2-Audio-7B-Instruct` @
`0a095220c30b7b31434169c3086508ef3ea5bf0a`) are now GPU load-tested — 4-bit NF4, **peak 6.54 GiB**
on the 4090.

| metric | value |
|---|---|
| **strict targeted attack-success** | **0.0133 (2/150)** |
| any restricted tool-call induced | **0.72 (108/150)** |
| by tool (strict / any) | transfer_money 0.00 / 0.70 · ssh_exec 0.042 / 0.79 · open_door 0.00 / 0.67 |
| action histogram | ssh_exec 102 · normal_text 42 · open_door 6 · transfer_money 0 |

**Interpretation — the most nuanced finding of the phase.** The *specific* white-box attack does
**not** transfer: 1.3% targeted success is at the noise floor. But the triggers are far from inert
on a different model — they induce a restricted tool call **72%** of the time, overwhelmingly
collapsing Qwen onto an **`ssh_exec` attractor** (102/150) regardless of what tool the trigger
targeted on SALMONN (e.g. `transfer_money` triggers yield 0% targeted success yet 70% any-restricted,
emitting ssh_exec instead). So white-box *specificity* is model-bound.

> **Corrected 2026-10-17.** The original reading here ("a generic jailbreak pressure partially
> survives the model swap ... a cross-model hazard") is withdrawn. The later emission-null control
> (`out/phase8/actuation_nulls_lam0.json`, `actuation_nulls_lam1.json`) showed that on SALMONN the
> instruction prompt already elicits a restricted tool call on clean audio (0.987) and on random
> noise (1.000). No Qwen clean or noise baseline was measured, so the 72% is consistent with the
> same prompt saturation and is not evidence of transfer; only the strict targeted rate (1.3%) is
> interpretable, and it shows the attack does not port.

---

## What Phase 8 adds to the Phase-7 story

Phase 7 established the confirmatory result (DEFENDER_LOSS, 0/4, FPR-driven; detector holds at
λ ≤ 0.1 and breaks at λ ≥ 1 on the pre-committed reframe). Phase 8 bounds *deployability*:

- The attack is **channel-fragile** — neither lossy codec nor over-the-air capture leaves it
  working, which usefully narrows the real-world threat to clean digital-injection paths.
- The detector is **not channel-robust** — it is incidentally helped by codec but disabled by
  over-the-air, so a clean-trained passive detector is not deployable against acoustic capture
  without OTA-augmented training.
- The attack is **not a portable exploit** across models, but the trigger *class* transfers as a
  generic jailbreak pressure.

All exploratory, all on non-sealed data, all reproducible from the pinned code. Nothing here
touches or depends on the burned held-out set.
