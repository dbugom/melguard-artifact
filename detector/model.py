"""MelGuard AST detector g_D (Phase 4).

Backbone: the pinned full AST (SPEC-CHANGE-006 — `MIT/ast-finetuned-audioset-10-10-0.4593`
@ f826b80d, `ASTForAudioClassification`, hidden 768, AudioSet-pretrained). We load
the encoder (`ASTModel`), freeze it except the last N transformer layers + the final
LayerNorm, and attach a retrained binary head:

    mean-pool over time -> Linear(768 -> 64) -> GELU -> Dropout(0.1) -> Linear(64 -> 1)

(brief §5.3, with the first Linear's input dim 192->768 per SPEC-CHANGE-006). Every
dimension / freeze depth is read from env/MODEL_PINS.lock.json — nothing hardcoded.
Trained with BCE-with-logits; `predict_proba` applies the sigmoid at inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from melguard.locks import model_pins


def detector_pins() -> dict:
    return model_pins()["models"]["detector_backbone"]


class MelGuardDetector(nn.Module):
    def __init__(
        self,
        ast_path: str | None = None,
        revision: str | None = None,
        freeze_except_last: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        from transformers import ASTModel

        d = detector_pins()
        ast_path = ast_path or d["hf_repo"]
        revision = revision or d["hf_revision"]
        hidden = int(d["hidden_size"])

        self.backbone = ASTModel.from_pretrained(ast_path, revision=revision)
        self.hidden = hidden
        self._freeze(freeze_except_last)

        # brief §5.3 head, input dim = pinned hidden_size (768) per SPEC-CHANGE-006.
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _freeze(self, keep_last: int) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        layers = self.backbone.encoder.layer
        for layer in layers[len(layers) - keep_last:]:
            for p in layer.parameters():
                p.requires_grad = True
        # final LayerNorm stays trainable (brief §5.3: "frozen except last 2 layers")
        for p in self.backbone.layernorm.parameters():
            p.requires_grad = True

    # optimizer param groups (backbone vs head get different LRs)
    def trainable_backbone_params(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def head_params(self):
        return list(self.head.parameters())

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """input_values: (B, max_length, num_mel_bins) from ASTFeatureExtractor.
        Returns raw logits (B,) — use BCEWithLogitsLoss for training."""
        hidden_states = self.backbone(input_values).last_hidden_state  # (B, seq, H)
        pooled = hidden_states.mean(dim=1)                              # mean-pool over time
        return self.head(pooled).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, input_values: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(input_values))
