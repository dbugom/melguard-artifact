"""v1 (hardened, lambda=0 only) vs v2 (adaptive, lambda=0+1) on the dev triggers
(seed 31337 — NOT gradient-trained by either; used only for tau_op). Key question:
does v2 catch the lambda=1 attacks that beat v1? Plus v2 per-language benign FPR."""
import numpy as np, torch
from detector.data import build_or_load_features
from detector.model import MelGuardDetector
from detector.infer import predict_all, select_tau_op

dev = build_or_load_features("dev", "/workspace/data/features")
labels = dev["labels"].numpy()
dsets = np.array([str(d) for d in dev["datasets"]])
langs = np.array([str(l) for l in dev["langs"]])
benign = labels == 0
lam0 = dsets == "phase5_lam0"
lam1 = dsets == "phase6_lam1"

def load(w):
    m = MelGuardDetector().to("cuda:0")
    m.load_state_dict(torch.load(w, map_location="cuda:0", weights_only=True)); m.eval(); return m

v1 = load("out/phase6/detector_hardened/detector_gd_best.pt")
v2 = load("out/phase6/detector_adaptive/detector_gd_best.pt")
p1, _ = predict_all(v1, dev, "cuda:0")
p2, _ = predict_all(v2, dev, "cuda:0")
t1 = select_tau_op(p1[benign], 0.02)
t2 = select_tau_op(p2[benign], 0.02)

print(f"[cmp] tau_op  v1={t1:.5f}  v2={t2:.5f}")
print(f"[cmp] dev benign FPR  v1={100*(p1[benign]>t1).mean():.1f}%  v2={100*(p2[benign]>t2).mean():.1f}%")
for name, mask in [("lambda=0 dev triggers", lam0), ("lambda=1 dev triggers", lam1)]:
    n = int(mask.sum())
    print(f"[cmp] {name} (n={n}):")
    print(f"[cmp]    v1 CAUGHT {100*(p1[mask]>t1).mean():.0f}%  (median p_det {np.median(p1[mask]):.4f})")
    print(f"[cmp]    v2 CAUGHT {100*(p2[mask]>t2).mean():.0f}%  (median p_det {np.median(p2[mask]):.4f})")
print("[cmp] v2 per-language benign FPR:", {lg: round(100*(p2[benign & (langs==lg)]>t2).mean(), 1)
      for lg in ("en", "es", "zh-CN") if (benign & (langs==lg)).sum()})
