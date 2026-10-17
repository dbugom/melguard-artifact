"""Phase-4: train the baseline AST detector g_D.

Train set  = benign_train (LibriSpeech) + public_train (JALMBench adversarial).
Dev set    = benign_val + public_dev (used only to track AUC / select tau_op).
Loss       = BCE-with-logits + class-balanced pos_weight (n_neg/n_pos).
Optimizer  = AdamW, lr 3e-5 (backbone last-2 layers) / 3e-4 (head), cosine warmup.
Epochs/batch and all hyperparameters come from env/MODEL_PINS.lock.json.

Weights are written to out/phase4/ (git-ignored .pt; canonical copy -> R2/volume).
This is the baseline detector; the adaptive-adversarial retrain is Phases 5-6.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from detector.data import build_or_load_features
from detector.infer import predict_all
from detector.model import MelGuardDetector
from melguard.locks import model_pins, number_lock


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main() -> int:
    tr = model_pins()["models"]["detector_backbone"]["train"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=int(tr["epochs"]))
    ap.add_argument("--batch", type=int, default=int(tr["batch_size"]))
    ap.add_argument("--limit", type=int, default=None, help="smoke: cap clips per split")
    ap.add_argument("--cache-dir", default="/workspace/data/features")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="out/phase4")
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    args = ap.parse_args()

    seed = int(number_lock()["attack_seeds"][0])  # 1729
    set_seed(seed)

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = out_dir / "detector_gd.pt"
    best_weights = out_dir / "detector_gd_best.pt"

    print(f"== Phase-4 detector train == epochs={args.epochs} batch={args.batch} "
          f"limit={args.limit} seed={seed}")
    train = build_or_load_features("train", args.cache_dir, limit=args.limit)
    dev = build_or_load_features("dev", args.cache_dir, limit=args.limit)

    model = MelGuardDetector().to(args.device)
    X, y = train["input_values"], train["labels"].float()
    n = len(X)
    n_pos = int(train["labels"].sum())
    n_neg = n - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=args.device, dtype=torch.float32)
    print(f"[train] {n} clips  pos(adv)={n_pos}  neg(benign)={n_neg}  pos_weight={pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW([
        {"params": model.trainable_backbone_params(), "lr": float(tr["lr_backbone"])},
        {"params": model.head_params(), "lr": float(tr["lr_head"])},
    ])
    steps_per_epoch = math.ceil(n / args.batch)
    total_steps = steps_per_epoch * args.epochs
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(args.warmup_frac * total_steps), total_steps
    )

    gen = torch.Generator().manual_seed(seed)
    log = []
    best_auc = -1.0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        running = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            xb = X[idx].to(device=args.device, dtype=torch.float32)
            yb = y[idx].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(idx)

        probs, dl = predict_all(model, dev, args.device)
        dev_auc = float(roc_auc_score(dl, probs))
        rec = {
            "epoch": epoch + 1,
            "train_loss": round(running / n, 5),
            "dev_auc": round(dev_auc, 4),
            "lr_head": scheduler.get_last_lr()[1],
        }
        log.append(rec)
        print(json.dumps(rec), flush=True)

        torch.save(model.state_dict(), weights)
        if dev_auc > best_auc:
            best_auc = dev_auc
            torch.save(model.state_dict(), best_weights)

    (out_dir / "train_log.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in log), encoding="utf-8"
    )
    print(f"\n== train done == best_dev_auc={best_auc:.4f} "
          f"weights={weights.name} best={best_weights.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
