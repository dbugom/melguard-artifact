# Vendored third-party code — `salmonn/upstream/`

The official SALMONN inference code, copied **verbatim** from the checkpoint repo
so the "official code path" is reproducible without a network fetch at run time.

- **Source:** `https://huggingface.co/tsinghua-ee/SALMONN`
- **Revision:** `e0e903b8a07f623616d1b950b8846b4535b76666` (the pin in `env/MODEL_PINS.lock.json`)
- **License:** Apache-2.0 (`salmonn/upstream/LICENSE`). Third-party component licenses
  (Vicuna, Whisper, LAVIS/MiniGPT-4/VideoLLaMA Q-Former, BEATs) are retained in the
  upstream tree.
- **Files:** `model.py`, `cli_inference.py`, `requirements.txt`, `LICENSE`,
  `beats/` (BEATs encoder), `qformer/Qformer.py` (window-level Q-Former).

## Modifications to the vendored files

Kept to the minimum needed to run the pinned upstream on our pinned stack
(transformers 4.46.3 vs. upstream's 4.28). Each edit is marked `MELGUARD-PATCH`.

| File | Change | Why |
|---|---|---|
| `qformer/Qformer.py` | `from transformers.file_utils import ModelOutput` → `from transformers.utils import ModelOutput` | `transformers.file_utils` was removed; `ModelOutput` moved to `transformers.utils`. |
| `qformer/Qformer.py` | `apply_chunking_to_forward`, `find_pruneable_heads_and_indices`, `prune_linear_layer` imported from `transformers.pytorch_utils` (was `transformers.modeling_utils`) | These helpers moved to `transformers.pytorch_utils`. |

`model.py` is **not** modified in place. Our loader does not import it; instead
`salmonn/model.py` is an adaptation (`MelGuardSALMONN`) derived from it, with the
spec-mandated changes (4-bit NF4 Vicuna, fp16 encoders, quantised-base-safe device
placement, deterministic greedy generate). The verbatim upstream `model.py` is
retained only as the reference for that adaptation.

## How the vendored packages are imported

`beats/` and `qformer/` use **absolute** imports (`from beats...`, `from qformer...`).
`salmonn/__init__.py` (and `salmonn/model.py`) put `salmonn/upstream` on `sys.path`
so they resolve as top-level packages, leaving the vendored files otherwise untouched.
