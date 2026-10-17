# Deviations from pre-registration

> **Status: complete through Phase 8.** Paper-ready prose for the "Deviations from
> pre-registration" section. Audited against the verbatim registration export in
> `out/YFI-17-rq3-success-criteria.md` (OSF `j5x2e`, registered 2026-07-05, public,
> `reviews_state=accepted`). Every claim is traceable to a `SPEC-CHANGE` entry in
> `out/SPEC-CHANGE-LOG.md`. Deviations 1–4 were fixed before any held-out access; 5–7 were
> ratified before the unseal (SPEC-CHANGE-014, 2026-07-23) and are recorded here with the
> sealed numbers now known; 8 is a formula-sign correction of the registered objective
> (SPEC-CHANGE-012, ratified 2026-07-17).

---

## Deviations from pre-registration

This study was pre-registered on the Open Science Framework (osf.io/j5x2e) on 2026-07-05,
before any adversarial trigger generation and before any access to the sealed held-out set.
We report **all** deviations from the registered plan below, each with its reason and its
consequence for inference. Deviations 1–4 were identified and remediated before any held-out
access. Deviations 5–7 are interpretations of genuine ambiguities or internal contradictions
in the registered text; each was resolved in writing and time-stamped **before** the single
unseal, so that no choice could be informed by a held-out number.

### What did not change

The pre-registered core is unchanged: (i) the security predicate
`secure(g_D, λ_det) := AUC ≥ 0.90 ∧ FPR@τ_op ≤ 0.02 ∧ actuation_rate ≤ 0.05`, evaluated with
the registered CI gates (AUC 95% bootstrap-CI lower bound excluding 0.90; FPR 95% Wilson-CI
upper bound ≤ 0.025; actuation as a point threshold); (ii) the pre-committed outcome rule —
positive iff 4/4 conditions secure, partial-positive iff exactly 3/4 (naming the breaking
condition), defender-loss iff ≤ 2/4; (iii) the four adversary-strength conditions
λ_det ∈ {0, 0.1, 1.0, 10.0}; (iv) the analysis machinery — a 1,000-iteration paired bootstrap
resampling carrier-utterance clusters (the coverage-correct realization of the registered
"stratified over carrier-utterance-id × language" design; see Deviation 10), one-sided tests
against the 0.90 floor and the 0.02/0.025 ceilings, Holm–Bonferroni across the four conditions
at family-wise α = 0.05, BCa for AUC and Wilson for FPR; (v) the carrier design — 25 fixed held-out carrier utterances × 3
languages, held constant across all four λ_det and all three attack seeds; (vi) the seal —
τ_op was selected on dev and frozen before any held-out access, and the held-out set was
evaluated exactly once; and (vii) the dual-use guardrails — no working trigger waveform in the
paper, supplement, or release, no live tool endpoints, and coordinated disclosure preceding
publication.

### Deviation 1 — Detector backbone: registered "AST-tiny", deployed the full AST

**Registered.** The primary hypothesis and the transformations section describe an "AST-tiny"
Mel-spectrogram detector.

**Actual.** The backbone is the full Audio Spectrogram Transformer
(`MIT/ast-finetuned-audioset-10-10-0.4593`, hidden size 768, ≈86 M parameters), not a
192-dimensional ≈22 M "AST-tiny".

**Reason.** The "AST-tiny" label was inherited from a design-brief description that
contradicted the model checkpoint pinned in the same locked specification. The pinned AudioSet
checkpoint *is* the full AST; no "AST-tiny" variant of it exists. We resolved in favour of the
pinned checkpoint — the artifact the study actually locks, hashes, and reproduces — before any
held-out evaluation (SPEC-CHANGE-006).

**Consequence for inference.** The deployed detector has *more* capacity than registered. This
is conservative with respect to our (defender-loss) headline: a failure to meet the bar is
*stronger* evidence against passive Mel-spectrogram gateway detection, because it cannot be
attributed to an under-parameterised model. It forecloses only a claim we do not make — we
nowhere claim a ≈22 M-parameter detector suffices, and we do not describe the detector as
"tiny". Backbone capacity is reported explicitly in §Methods.

### Deviation 2 — Benign training composition: registered LibriSpeech-only, used multi-corpus

**Registered.** Benign training/dev clips from LibriSpeech (train-clean-100 + dev-clean);
FLEURS "used only in robustness sub-studies".

**Actual.** Benign training/dev clips drawn from **LibriSpeech + CommonVoice (train split) +
FLEURS (train/validation splits)**, language-balanced across en/zh-CN/es.

**Reason.** The registered composition — a single English read-speech benign corpus against
adversarial positives from a different corpus — induces a **corpus-provenance shortcut** rather
than trigger detection. We verified this empirically before any held-out access: the baseline
detector reached dev AUC = 1.00 at τ_op = 0.0013, yet benign false-positive rate on the
non-sealed FLEURS control reached **9.3% (English)** — the decision boundary encoded "is this
LibriSpeech?" rather than "does this contain a trigger?". The registration's data answer was
also internally inconsistent with the registered research question, which specifies a detector
"trained on benign LibriSpeech/CommonVoice/FLEURS"; we resolved toward the research question
(SPEC-CHANGE-007).

**Leakage control.** The additional benign data is drawn exclusively from the CommonVoice
**train** split and FLEURS train/validation, disjoint by construction from the sealed held-out
sets (CommonVoice validated/test). No held-out clip entered training or threshold selection.

**Consequence for inference.** After remediation, per-corpus dev FPR is balanced at ≈2%
(CommonVoice 0.021 / FLEURS 0.016 / LibriSpeech 0.023) and the FLEURS-English control FPR falls
from 9.3% to 5.3%. The change strengthens construct validity. A residual English-vs-other
asymmetry remains and is confirmed on the sealed set (Deviation-independent finding, §Results:
en 2.70% / es 2.04% / zh-CN 1.80% held-out benign FPR).

### Deviation 3 — Class-balance ratio: registered ≈1:11.8, actual 1:20.8

**Registered.** Class-balanced re-weighting at positive:negative ≈ 1:11.8 during training.

**Actual.** 1:20.8 — the true ratio of the training split under the corrected composition
(Deviation 2); the registered figure was an estimate carried from an earlier draft.

**Consequence for inference.** None. Re-weighting is applied only during detector training, not
at analysis; the reported outcomes (AUC, FPR, actuation) are untransformed.

### Deviation 4 — Held-out data manifest re-hashed after registration

**Registered.** "Inclusion is governed by a locked SHA-256 data manifest"; the held-out set is
sealed and "touched exactly once at write-up".

**Actual.** The data manifest was regenerated after registration; every clip's bytes — and
therefore every hash — changed.

**Reason.** The original pipeline invoked resampling only for corpora whose native sample rate
differed from 16 kHz, so a resampler anti-aliasing signature correlated with the adversarial
label *and* was carried by the sealed benign CommonVoice clips — a confound that would have
inflated held-out FPR. We routed every corpus through a single uniform decode-and-resample path
(SPEC-CHANGE-008), which re-encodes the same underlying recordings.

**Seal integrity.** The held-out **selection is provably unchanged**: the same seeded RNG,
caps, and round-robin select the same rows, and the held-out draws use a separate RNG instance.
The re-encode changes the representation of the same recordings, not which recordings are held
out, and the held-out set was not evaluated before the single pre-registered evaluation. Both
pre- and post-remediation manifest hashes are reported for audit.

### Deviation 5 — Per-condition positive class: registered wording implies 425/λ; we use 75/λ

**Registered.** Two registration passages are in arithmetic conflict. The study-design section
(344-40) states each condition is evaluated on "425 adversarial clips **optimized at that
λ_det**". The sample-size section (344-51) states N = 425 = 300 in-house generated (attack-seed
s3) + 125 public-held, and that the generated clips come from a cube of 4 λ × 3 seeds × 3
languages × 25 prompts = 900, split 300/300/300 into train/dev/held.

**The contradiction.** The cube splits by attack seed, so the held split contains
4 λ × 1 seed × 3 languages × 25 = **300 generated clips across all four λ — i.e. 75 per λ**.
"425 optimized at that λ_det" would require 1,700 λ-specific generated clips; the entire cube
is 900. The two statements cannot both hold, and the 125 public-held clips (JALMBench attacks)
were never optimized at any λ, so they cannot be "optimized at that λ_det" at all.

**Actual.** Our **primary** analysis defines each condition's positive class as the **75
λ-specific generated held clips only** (`gen-only`). We report a **pre-specified secondary**
analysis (`gen+public`, N = 198/λ = 75 generated + 123 shared public) for completeness
(SPEC-CHANGE-014, D1).

**Reason.** The registered outcome (344-66) is `AUC(g_D | adv@λ_det, benign_held)` — the
positive class is conditioned on λ_det. The public clips are not λ-specific, so pooling them
contaminates the single-factor λ design. Moreover, because AUC is a linear mixture over
positives, `AUC_{gen+public}(λ) = (75·AUC_gen(λ) + 123·AUC_pub)/198`, which **attenuates the
between-condition λ contrast by 75/198 ≈ 0.38** — a ≈2.6× compression of the effect the study
exists to measure. We verified this identity numerically before the unseal (mixture prediction
matched to machine precision; attenuation 0.3788 = 75/198 exactly), and the sealed run confirms
it: the λ-spread in AUC is 0.092 under `gen-only` versus 0.035 under `gen+public`.

**Consequence for inference.** We disclose 344-40's "425 … optimized at that λ_det" as an
**arithmetic erratum** — it conflates the pooled held-out inventory (425 = 300 + 125) with the
per-condition positive class. The `gen-only` primary reading is smaller (75/λ) and therefore
has **wider confidence intervals**; we do not hide this — see the statistical-limitation note
below on finite-sample CI coverage. The choice does not change the confirmatory outcome:
both readings return defender-loss 0/4.

### Deviation 6 — Multiplicity correction: the registered Holm clause is internally ambiguous

**Registered.** 344-77 defines a condition as passing iff three CI-gated tests hold jointly at
the **95%** level, and *in the same sentence* adds "with Holm–Bonferroni correction across the
four [conditions] at family-wise α = 0.05". 344-71 additionally asks for a "Holm-corrected
p-value" per condition.

**The ambiguity.** Both cannot bind simultaneously. If Holm corrects the composite per-condition
hypothesis (the literal reading of "across the four conditions"), the effective FPR ceiling
tightens from k ≤ 100 to k ≤ 97 false positives at n = 5,000 and the registered "95% Wilson
upper ≤ 0.025" is no longer the operative pass rule. The registration does not say which
statistic Holm corrects.

**Actual.** We compute and report **all three** internally-consistent readings on every run —
(a) `registered`: the 95% CI gates with no multiplicity adjustment; (b) `holm_auc` (our
pre-committed primary): Holm applied to the AUC test across the four λ, with FPR and actuation
un-adjusted; (c) `holm_composite`: Holm on the intersection-union composite p-value. The primary
is `holm_auc` (SPEC-CHANGE-014, D3).

**Reason.** We honour the CI sentence and correct only the **λ-varying** component (AUC),
because FPR is **λ-invariant** — it is a single measurement on one benign set at one frozen
τ_op, reported four times, so a ×4 multiplicity penalty on four identical copies controls no
real error rate. We state plainly that the *literal* reading of "across the four conditions"
points at `holm_composite`, and that we depart from it with the stated reason.

**Consequence for inference.** None to the headline: the FPR failure is large enough that all
three readings return **0/4, defender-loss**, identically. The ambiguity is therefore moot for
our result, but we disclose it and report all three for auditability.

### Deviation 7 — Held-out adversarial N: registered 425, actual 423

**Registered.** N = 425 = 300 generated + 125 public-held.

**Actual.** The de-duplicated public-held split contains **123** clips, not 125, so the primary
adversarial total is **423**. The registration itself hedges the public corpus as "~825
post-dedup, split ~575/~125/~125", so the realized 123 is within the registered tolerance.

**Consequence for inference.** None. No gate or threshold depends on the value 425; the count is
reported for completeness (SPEC-CHANGE-014, D5).

### Deviation 8 — Adversary objective: the registered sign is anti-evasion

**Registered.** The manipulated-variables entry (344-58) writes the adversary's objective as
`max over δ of E[ L_actuate(H(δ⊕x_b)) − λ_det · log(1 − g_D(δ⊕x_b)) ]` — with a **minus** sign
on the detector-evasion term.

**Actual.** We optimize the evasion-correct sign, `max over δ of E[ L_emit + λ_det · log(1 − g_D) ]`
(equivalently, the code minimizes `CE − λ_det·log(1 − p_det)`).

**Reason.** `g_D` is the probability that the input is adversarial, and the gateway rejects
`g_D > τ_op`, so *evasion* means driving `g_D` **down**. The registered term `−λ·log(1 − g_D)` is
an *increasing* function of `g_D`, so **maximizing** it drives `g_D` **up** — toward maximal
detectability, the opposite of evasion. The literal registered sign is therefore anti-evasion; it
contradicts the standard white-box detector-aware threat model the registration otherwise
describes (an attacker that *minimizes* detectability). We treat this as a formula-sign
typographical error in the registered objective and correct it (SPEC-CHANGE-012, ratified
2026-07-17). Verified two ways before use: a first-principles gradient analysis and an independent
re-derivation, both concluding the literal sign is anti-evasion.

**Empirical confirmation.** The evasion-correct sign was checked to actually reduce detectability:
on the highest-scoring public (JALMBench) positive, minimizing the evasion penalty drove
`p_det` from `0.9999` to `~0` in about five PGD steps; and the Phase-6 sweeps observed the
predicted detectability collapse across all four λ (held-clip evasion rising 0 → 0 → 84% → 100%
in the Phase-7 eval).

**Consequence for inference.** None to the headline. FPR is λ-invariant and is the binding
failure (2.18% vs the 0.02 ceiling), so the confirmatory outcome is defender-loss 0/4 regardless
of the evasion-term sign; the sign only affects how efficiently the λ>0 attacks evade, which the
result already shows they do. Disclosed as a formula-sign correction of the registered objective.

---

### Deviation 9 — Triage conditions re-scored, not independently replicated

**Registered.** The held-out set is "touched exactly once at write-up"; each condition is
evaluated on its own held (attack-seed s3) positives.

**Actual.** The compute triage (SC-013) generated λ ∈ {0.1, 10} at one-third scale with a
**single** attack seed (s3). For those two conditions the Phase-7 held clips are therefore the
same clips generated in Phase 6: they are **re-scored** under the frozen detector, not
independently replicated. The full-scale λ ∈ {0, 1} conditions use fresh seed-s3 held triggers
distinct from the dev clips.

**Consequence for inference.** Bounded. Re-scoring cannot convert a failed condition to secure
(the binding rule keeps a failed condition failed), and the headline is FPR-bound and
λ-invariant, so 0/4 is unaffected. It does weaken the λ-varying **AUC** narrative for the two
triage conditions, which we now state explicitly in the paper (§Results, "On replication").

### Deviation 10 — Bootstrap resampling unit: the registered strata are singletons

**Registered.** "Paired stratified bootstrap over carrier-utterance-id × language strata."

**Actual.** Those strata are singleton cells (one clip per carrier × language), for which
within-stratum resampling is degenerate: it returns each cell's single value with zero variance,
producing an invalid interval ≈ 9.2× too narrow (empirically ~43% type-I error against the
0.90 floor). We instead resample the **carrier-utterance cluster** — the coverage-correct
realization of the registered *paired* design — which widens every interval and restores
near-nominal coverage (simulated 92.7% at true AUC 0.90, n_pos = 75).

**Consequence for inference.** Conservative. The correction *widens* the AUC intervals; it makes
the AUC gate harder to pass, never easier, and does not change the FPR-bound 0/4 outcome.

---

## Pre-specified additions (not deviations)

These were fixed in writing before the unseal (SPEC-CHANGE-014) and add reporting without
altering the registered pass/fail:

- **FPR CI-ceiling locked.** 344-77's decision-determining Wilson-upper ceiling (0.025) existed
  only as a source constant; we added `headline_bar.fpr_ci_upper_max = 0.025` to the locked
  quantities with a regression test (SC-014, D4). No value changed.
- **Threshold-transfer diagnostic (firewalled).** Because the sealed benign-FPR set is 100%
  CommonVoice while τ_op was frozen on a three-corpus dev distribution, we pre-specified (before
  the unseal) a second, corpus-matched threshold (selected on dev CommonVoice only, at the same 0.02 target)
  as an **explanatory diagnostic**. It is firewalled in code and prose: it may explain an FPR
  failure as a corpus-mismatch artifact but may **never** promote a failed condition to secure.
  Both thresholds are computed from the one unseal.
- **Per-condition analysis beyond the registered defender-loss reframe (344-2), pre-specified.**
  We pre-specified before the unseal (SC-014, 2026-07-23) that AUC and actuation would be reported
  and interpreted per condition even under a global FPR failure — going beyond the registered
  reframe — with the binding rule that a failed condition stays failed.
- **Exploratory benign decomposition.** Per-language, per-corpus, per-voice-type and per-SNR
  FPR (344-44/344-62 register these as held-out subsetting variables) and a precision check
  reading FPR on the benign-AUC set (doubling the benign sample to 10,000). Exploratory
  (344-83); the confirmatory FPR gate remains the benign-FPR set alone.

---

## Statistical limitation (reported, not a deviation)

At the primary per-condition size n_pos = 75, the BCa AUC interval undercovers in finite
samples. Simulated at a true AUC of exactly 0.90 (n_neg = 5,000, 10,500 replications) the
nominal 95% interval covers **92.7%** (and the AUC gate's one-sided error is 3.71% against a
nominal 2.5%), versus **95.5%** at n_pos = 198. This is standard finite-sample BCa behaviour,
not an implementation defect (the interval matches `scipy.stats.bootstrap(BCa)` to ≤ 0.0021).
We therefore do **not** claim exact 95% coverage or exact α = 0.05 error control for the AUC
gate at n_pos = 75, and we report it alongside the `gen+public` reading (n_pos = 198), which
recovers ≈95% coverage.

**Three hardening items deferred to the camera-ready** (none can change the FPR-bound 0/4, and
each direction is known a priori): (a) **speaker-clustered negatives** — the 5,000 benign clips
are currently resampled independently rather than clustered by CommonVoice `client_id`;
clustering can only *widen* the FPR/AUC intervals, so it cannot rescue the FPR gate (which fails
on both the point estimate 0.0218 > 0.02 and the Wilson upper 0.0262 > 0.025) and only sharpens
it; (b) a **paired ΔAUC / trend test** over the shared carrier structure, to formalize the
qualitative AUC decline; and (c) **bootstrap draws** raised from B = 1,000 to B ≥ 50,000 to
de-noise the extreme Holm-level tail quantiles. The re-analysis is scripted in
`robustness/phase8_sensitivity.py` (a post-hoc sensitivity pass on the already-unsealed scores —
it does not re-open the seal); it needs the frozen detector scores regenerated in a torch
environment, plus the recovered `client_id` map for (a).

## Robustness sub-studies ran on non-sealed data (consistent with the registration)

The Phase-8 robustness sub-studies (codec, over-the-air, cross-language, Qwen2-Audio transfer;
344-83) were evaluated on the **non-sealed** FLEURS control set and dev-seed (s31337) triggers,
not the sealed held-out set. This is consistent with 344-47 ("FLEURS … used only in robustness
sub-studies") and 344-83 (the Qwen transfer reports "attack-success rate only, not detector
AUC"). We disclose that these results are therefore on a different benign distribution and a
different attack seed than the headline and are exploratory and non-confirmatory. The registered
per-language sub-AUC / bias audit, which 344-53 sizes on the sealed benign set (~1,667 negatives
per language), was folded into the single unseal to avoid a second touch of the sealed set.
The registered cross-language sub-study ("leave-one-language-out") was realized as a
per-language breakdown of the same all-language detector; no language was held out of training.
The paper discloses this substitution.

---

## Implementation details not fixed by the registration

Reported in §Methods, not as deviations: the harness instruction wording; the attacker's target
argument strings (the registration fixes the restricted-tool set and the actuation predicate,
not the arguments); the PGD step size α (5,000 steps and the sign-gradient ℓ∞-clipped update are
fixed, α is not); the per-step batch size (a throughput choice, results invariant to it); the
standardised carrier length for batching; and the software channel implementations for the
Phase-8 sub-studies (ffmpeg libopus/libvorbis for codec, an image-source RIR for over-the-air).

---

## Reviewer-facing summary (one paragraph)

We report ten deviations. Four (1–4) were corrections made before any held-out access: two
resolve internal contradictions in the registered plan itself (the "AST-tiny" label versus the
pinned full-AST checkpoint; the benign-training composition versus the registered research
question), one is a consequential arithmetic correction, and one is a data-encoding fix that
removed a label-correlated resampling confound while provably preserving the held-out selection.
Three (5–7) are interpretations of genuine ambiguities or arithmetic conflicts in the registered
text — the per-condition positive-class size, the multiplicity-correction target, and the exact
public-held count — each resolved and time-stamped before the single unseal. The eighth is a
formula-sign correction of the registered adversary objective (344-58), whose literal minus sign
is anti-evasion; we optimize the evasion-correct sign, consistent with the threat model the
registration otherwise describes. Two further analysis choices are disclosed at unseal: (9) the
single-seed triage makes λ ∈ {0.1, 10} re-scored rather than independently replicated, and (10)
the registered singleton strata force a coverage-correct switch to carrier-cluster resampling
that widens every interval. None alters the pre-registered predicate, conditions, decision
rule, or seal, and none changes the confirmatory outcome, which is **defender-loss (0/4) under
all three multiplicity readings** — deviation 8 in particular is inert to the headline because
FPR (the binding failure) is λ-invariant. The two substantive early deviations (1, 2) each make
the reported test more conservative or more valid.
