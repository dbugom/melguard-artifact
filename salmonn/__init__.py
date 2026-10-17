"""MelGuard SALMONN-13B bring-up (Phase 2) — the white-box target f_S.

Wraps the official ``tsinghua-ee/SALMONN`` inference code (vendored verbatim under
``salmonn/upstream``, Apache-2.0) with a lock-driven loader:
  * Vicuna-13B backbone in 4-bit NF4 (bitsandbytes), Whisper+BEATs encoders fp16;
  * every component pinned by HF revision / SHA-256 in ``env/MODEL_PINS.lock.json``;
  * a deterministic (greedy) forward-only smoke test.

Importing this package puts ``salmonn/upstream`` on ``sys.path`` so the vendored
absolute imports (``from beats...``, ``from qformer...``) resolve. Heavy deps
(torch/transformers/bitsandbytes) are imported lazily by the submodules, so
``import salmonn`` itself stays cheap.
"""
from __future__ import annotations

import os
import sys

_UPSTREAM = os.path.join(os.path.dirname(__file__), "upstream")
if os.path.isdir(_UPSTREAM) and _UPSTREAM not in sys.path:
    sys.path.insert(0, _UPSTREAM)

__all__ = ["_UPSTREAM"]
