# MelGuard — reproduction artifact

Artifact for *"MelGuard: A Pre-Registered Evaluation of a Passive Mel-Spectrogram Gateway
against White-Box Detector-Aware Sub-Second Audio-Jailbreak Triggers"*
(Mohammad Razavi and Khubayb Ahmed Al Naamany, Integrated Technologies, 2026).

- Pre-registration: https://osf.io/j5x2e (DOI 10.17605/OSF.IO/J5X2E), registered 2026-07-05.
- Detector weights (3 checkpoints, 345 MB each): Zenodo DOI 10.5281/zenodo.22862486.
- Preprint: arXiv (identifier added after posting). See `CITATION.cff`.

The study asks a single pre-registered question: can a passive AST Mel-spectrogram detector,
placed as a gateway in front of an audio LLM, catch sub-second adversarial triggers that a
**white-box, detector-aware** attacker is optimising specifically to slip past it? The decision
was made **once**, on a sealed evaluation set, under a CI-gated statistical rule fixed in
advance in the public, timestamped OSF pre-registration. The answer is a defender-loss (0 of 4
adversary-strength conditions secure): the detector holds against a weak detector-aware
adversary and is defeated by a strong one.

## Layout
- `detector/` — the AST Mel-spectrogram detector `g_D`: model, training, inference, the
  one-shot sealed-eval driver, and the CI-gated statistics (BCa bootstrap, Wilson CI,
  Holm–Bonferroni over the four adversary-strength conditions).
- `attack/` — the white-box, detector-aware joint-PGD attacker (differentiable front-end,
  surrogate loss, carrier pool).
- `harness/` — the tool-use harness. It has **no live endpoints** by construction: every
  restricted tool is a stub that raises if it is ever called, the harness measures by default
  and reaches a stub only through an explicit debug switch, and `tests/test_no_live_tools.py`
  asserts by AST inspection that no harness module imports a network, HTTP, or subprocess
  library.
- `robustness/` — the channel and transfer sub-studies (codec, over-the-air room impulse,
  cross-model transfer, the emission-null control, and the deferred statistical-sensitivity
  pass `phase8_sensitivity.py`).
- `data/` — data preparation and SHA-256 manifesting.
- `salmonn/` — loader for the target audio LLM (vendored upstream; see `salmonn/VENDORED.md`).
- `env/` — the locked reproduction pins (models, numbers, data manifest).
- `tests/` — governance + statistics tests.
- `out/` — the committed result JSONs, findings, the pre-registration export, and the
  one-shot seal record (`out/phase7/SEAL_BROKEN.json`). Every headline and table number in the
  paper regenerates from these. `out/phase7/HARNESS.sha256` records the thirteen
  evaluation-harness files exactly as they were when the sealed set was read, and
  `out/phase7/harness_frozen/` holds a byte-exact snapshot of all thirteen:
  `cd out/phase7/harness_frozen && sha256sum -c ../HARNESS.sha256` verifies 13/13. In their
  normal locations, nine are still identical (all evaluation code, both locks, the
  threshold-transfer diagnostic); four differ because they were edited after the sealed run for
  non-scientific reasons (three test modules gained `pytest.importorskip` guards for a
  lightweight CI job; `env/MODEL_PINS.lock.json` gained a "GPU load-tested" annotation on the
  Qwen2-Audio pin used only in the exploratory transfer sub-study). No evaluated value changed;
  see the snapshot's README.
- `docs/DEVIATIONS-FROM-PREREG.md` — the supplement: every deviation from the registration,
  its cause, and its consequence for inference. `docs/RESULTS.md` — the consolidated results.
- `paper/` — the paper source (`main.tex`, `refs.bib`), figures, and `make_figures.py`.

## Reproduce
1. Install the pinned stack (see `env/` for exact model + package pins).
2. `python -m data.prepare all` then `python -m data.manifests build`
3. `python -m pytest tests/`
4. `python -m paper.make_figures`
5. Download the detector weights from the Zenodo record above and verify them against its
   `SHA256SUMS`; `detector/infer.py --weights <file>` scores audio at the frozen thresholds
   recorded in `out/phase{4,6}/**/detector_metrics.json`.

## Dual-use
No working trigger waveforms are shipped. Triggers are distributed as SHA-256 hashes plus the
full optimizer configuration, so they regenerate under the pinned setup but cannot be replayed
from a supplied file. The frozen detector weights are released separately on Zenodo (they exceed
GitHub's file-size limit). A 90-day coordinated-disclosure window (Day 0 = 2026-07-17) preceded
this release. See `POLICY.md`.

## License
Code and documentation: Apache License 2.0 (`LICENSE`); the vendored SALMONN code under
`salmonn/upstream/` keeps its own Apache-2.0 license. Detector weights: CC BY-NC 4.0 (see the
Zenodo record); non-commercial use with attribution, contact the authors for other uses.
