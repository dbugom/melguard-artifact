"""Verify the hardened detector: (1) before/after trigger detection vs frozen
Phase-4, (2) generalization (dev triggers = seed 31337, NOT trained), (3) the
carrier-shortcut probe — score BARE carriers (no delta) to prove the detector
needs the perturbation, not just the carrier identity."""
import glob, json, os, torch, numpy as np, soundfile as sf
from detector.data import build_or_load_features
from detector.model import MelGuardDetector, detector_pins
from detector.infer import predict_all, select_tau_op
from attack.detector_frontend import ast_mel_config, detector_prob_grad
from data.config import DATA_ROOT
from transformers import ASTFeatureExtractor

DEV = build_or_load_features("dev", "/workspace/data/features")
labels = DEV["labels"].numpy()
dsets = np.array([d or "unk" for d in DEV["datasets"]])
langs = np.array([l or "unk" for l in DEV["langs"]])
gen = dsets == "generated_lam0"

def load(w):
    m = MelGuardDetector().to("cuda:0")
    m.load_state_dict(torch.load(w, map_location="cuda:0", weights_only=True))
    m.eval(); return m

mh = load("out/phase6/detector_hardened/detector_gd_best.pt")
mf = load("out/phase4/detector_gd_best.pt")
ph, _ = predict_all(mh, DEV, "cuda:0")
pf, _ = predict_all(mf, DEV, "cuda:0")
tau_h = select_tau_op(ph[labels == 0], 0.02)
tau_f = select_tau_op(pf[labels == 0], 0.02)

print(f"[verify] n_gen_triggers(dev,s31337)={int(gen.sum())}  tau_h={tau_h:.6f}  tau_f={tau_f:.6f}")
print(f"[verify] HARDENED  gen-trigger p_det: med={np.median(ph[gen]):.4f} min={ph[gen].min():.4f}  "
      f"DETECTED@tau_h={(ph[gen] > tau_h).mean()*100:.1f}%")
print(f"[verify] FROZEN-P4 gen-trigger p_det: med={np.median(pf[gen]):.6f} max={pf[gen].max():.6f}  "
      f"DETECTED@tau_f={(pf[gen] > tau_f).mean()*100:.1f}%")
for lg in ["en", "es", "zh-CN"]:
    m = gen & (langs == lg)
    if m.sum():
        print(f"[verify]   {lg}: n={int(m.sum())} hardened_detected={(ph[m] > tau_h).mean()*100:.0f}% "
              f"med_p={np.median(ph[m]):.3f}")

# --- carrier-shortcut probe: bare carriers (no delta) must NOT be flagged ---
fe = ASTFeatureExtractor.from_pretrained(detector_pins()["hf_repo"], revision=detector_pins()["hf_revision"])
cfg = ast_mel_config(fe)
pays = sorted(glob.glob("/workspace/melguard-backup/payloads/phase5_lam0_*_s31337_*.json"))
bare = []
for p in pays:
    try:
        d = json.load(open(p))
        cpath = d.get("carrier_meta", {}).get("path")
        if not cpath:
            continue
        fp = DATA_ROOT / cpath
        if not os.path.exists(fp):
            continue
        w, sr = sf.read(fp)
        if getattr(w, "ndim", 1) == 2:
            w = w[:, 0]
        x = torch.tensor(np.asarray(w, dtype=np.float32), device="cuda:0")
        with torch.no_grad():
            bare.append(float(detector_prob_grad(mh, x, mel_cfg=cfg).flatten()[0].item()))
    except Exception as e:
        print("[verify] carrier skip:", e)
bs = np.array(bare) if bare else np.array([np.nan])
print(f"[verify] BARE CARRIERS (no delta), hardened p_det: n={len(bare)} "
      f"med={np.nanmedian(bs):.6f} max={np.nanmax(bs):.6f} FLAGGED@tau_h={(bs > tau_h).mean()*100:.1f}%")

trig_caught = (ph[gen] > tau_h).mean()
bare_flagged = (bs > tau_h).mean()
verdict = ("GENUINE delta-detection (triggers caught, bare carriers pass)"
           if trig_caught > 0.8 and bare_flagged < 0.2 else "INVESTIGATE — possible carrier/artifact shortcut")
print(f"[verify] SHORTCUT VERDICT: {verdict}  (triggers_caught={trig_caught*100:.0f}%, bare_flagged={bare_flagged*100:.0f}%)")
