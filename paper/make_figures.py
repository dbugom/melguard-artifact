"""Generate the paper figures from the committed result JSONs (no hand-transcribed numbers).

Outputs vector PDF + PNG into paper/figures/. Self-contained (mathtext, no LaTeX needed).
  fig1_boundary        three stacked panels sharing lambda on x:
                         (a) the evade-and-emit predicate: sealed (Phase 7) vs generation-time (Phase 6)
                         (b) sealed detector AUC with BCa CIs and the 0.90 floor
                         (c) the decomposition: tool-call emission is flat, gate evasion climbs
  fig2_fpr_by_snr      (a) benign FPR by SNR bucket, (b) share of clips vs share of false positives
  fig3_channel         three stacked panels over the Phase-8 channels: AUC, benign FPR, actuation

Every plotted value is read from a file under out/. The one derived quantity, the lambda=0
final-delta emission rate, is computed from the committed per-step log out/phase5/pgd_sweep_lam0.jsonl
(the last checkpoint of each of the 225 runs), the same computation the paper reports as 182/225.

Design rules (kept deliberately plain for a two-column paper, grayscale-safe):
  one meaning per colour across all figures: blue = detector metric, orange = attack / failure,
  grey = thresholds and annotations; text is always dark grey, never the series colour;
  thresholds are thin solid grey lines labelled in the margin; legends sit in empty space or
  outside the data; markers carry a white ring so crossings stay legible; labels are selective.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#333333"; INK2 = "#666666"; RULE = "#9a9a9a"; SURFACE = "white"
C_DET = "#0072B2"       # detector metrics (Okabe-Ito blue)
C_ATT = "#D55E00"       # attack / failure (Okabe-Ito vermillion)
BLUES = ["#c6dbef", "#6baed6", "#08519c"]   # one-hue ramp: high -> mid -> low SNR (clean -> noisy)

FS_SMALL = 7.0          # smallest text anywhere (legends, annotations, margin labels)
W = 3.05                # design width in inches; the tight bbox with margin labels lands near the 3.33 in column
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": FS_SMALL, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": RULE, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.handlelength": 2.2, "legend.borderaxespad": 0.2,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def _load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote figures/{name}.pdf + .png")


def rule(ax, y, label, color=RULE, side="right", dy=0.0, lw=0.7):
    """A thin solid reference line with its label in the margin, never on the data."""
    ax.axhline(y, color=color, lw=lw, zorder=1)
    x = 1.0 if side == "right" else 0.0
    ax.annotate(label, xy=(x, y), xycoords=("axes fraction", "data"),
                xytext=(3 if side == "right" else -3, dy), textcoords="offset points",
                ha="left" if side == "right" else "right", va="center", fontsize=FS_SMALL,
                color=INK2, annotation_clip=False)


def panel_letter(ax, letter, title):
    ax.text(0.0, 1.03, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="left", va="bottom", color=INK)
    ax.text(0.085, 1.03, title, transform=ax.transAxes, fontsize=7.6, ha="left", va="bottom",
            color=INK2)


def lam0_final_delta_emission():
    """lambda=0 emission on the saved final delta: last checkpoint of each run, 225 runs."""
    runs = defaultdict(dict)
    for line in (ROOT / "out/phase5/pgd_sweep_lam0.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        runs[d["run_id"]][d["step"]] = d
    fin = [r[max(r)] for r in runs.values()]
    assert len(fin) == 225, f"expected 225 lambda=0 runs, found {len(fin)}"
    return sum(int(bool(x.get("act_l_actuate"))) for x in fin) / len(fin)


# ------------------------------------------------------------ fig1: boundary, AUC, decomposition
def fig1():
    p7 = _load("out/phase7/sealed_result.json")["views"]["primary"]["per_condition"]
    p7 = sorted(p7, key=lambda r: float(r["lambda"]))
    lam = [float(r["lambda"]) for r in p7]
    assert lam == [0.0, 0.1, 1.0, 10.0], f"unexpected condition set {lam}"   # the four locked conditions, ascending
    x = np.arange(len(lam))
    sealed = [r["actuation"] for r in p7]
    evade = [r["n_evaded"] / r["n_adv"] for r in p7]
    auc = [r["auc"] for r in p7]
    lo = [r["auc"] - r["auc_ci95"][0] for r in p7]
    hi = [r["auc_ci95"][1] - r["auc"] for r in p7]
    # Phase-6 generation-time attack-wins (best-over-trajectory) for lambda>0. lambda=0 ran with
    # no detector in the loop; its point is the saved-trigger rate against the hardened detector,
    # read from the sealed result (0/75 evaded on the held seed) — an inference, stated in the caption.
    lam0 = next(r for r in p7 if float(r["lambda"]) == 0.0)
    wins = {0.0: lam0["n_actuated_and_evaded"] / lam0["n_adv"]}
    for l, f in [(0.1, "out/phase6/sweep_lam0.1_result.json"),
                 (1.0, "out/phase6/sweep_lam1_result.json"),
                 (10.0, "out/phase6/sweep_lam10_result.json")]:
        d = _load(f)["attack_wins_best_over_traj"]
        wins[l] = d["rate"] if isinstance(d, dict) else float(d)
    winv = [wins[l] for l in lam]
    # tool-call emission on the saved final delta (generation-time): Phase 5 for lambda=0, Phase 6 otherwise
    emit = {0.0: lam0_final_delta_emission(),
            0.1: _load("out/phase6/sweep_lam0.1_result.json")["final_actuation"],
            1.0: _load("out/phase6/sweep_lam1_result.json")["final_actuation_alone"],
            10.0: _load("out/phase6/sweep_lam10_result.json")["final_actuation"]}
    emitv = [emit[l] for l in lam]

    fig, axes = plt.subplots(3, 1, figsize=(W, 4.9), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 0.85, 0.85], "hspace": 0.34})
    ax_a, ax_b, ax_c = axes
    xlim = (-0.35, 3.95)
    mk = dict(mec=SURFACE, mew=1.0, zorder=4)

    # (a) predicate
    ax_a.axvspan(xlim[0], 1.5, color=C_DET, alpha=0.05, lw=0)
    ax_a.axvspan(1.5, xlim[1], color=C_ATT, alpha=0.05, lw=0)
    ax_a.axvline(1.5, color=RULE, lw=0.6, ls=(0, (1.5, 2.5)), zorder=1)   # survives grayscale
    ax_a.text(0.3, 0.60, "detector\nholds", ha="center", va="center", fontsize=FS_SMALL, color=INK2)
    ax_a.text(2.7, 0.62, "detector defeated", ha="center", va="center", fontsize=FS_SMALL, color=INK2)
    rule(ax_a, 0.05, "bar 0.05", dy=-3)
    ax_a.plot(x, winv, "--", color=C_DET, lw=1.3, label="gen-time attack-wins (Phase 6)", zorder=3)
    ax_a.plot(x, winv, "s", color=C_DET, ms=4.5, mfc=SURFACE, mec=C_DET, mew=1.0, zorder=4)
    ax_a.plot(x, sealed, "-", color=C_ATT, lw=1.8, label="sealed predicate (Phase 7)", zorder=3)
    ax_a.plot(x, sealed, "o", color=C_ATT, ms=5, **mk)
    ax_a.annotate(f"{winv[-1]:.2f}", (x[-1], winv[-1]), xytext=(5, 2), textcoords="offset points",
                  fontsize=FS_SMALL, color=INK2, va="bottom")
    ax_a.annotate(f"{sealed[-1]:.2f}", (x[-1], sealed[-1]), xytext=(5, -2), textcoords="offset points",
                  fontsize=FS_SMALL, color=INK2, va="top")
    # head-room above the curves so the legend never touches data: the band y in [1.05, 1.45] is empty
    ax_a.set_ylim(-0.04, 1.48); ax_a.set_xlim(*xlim)
    ax_a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_a.set_ylabel("actuation rate")
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.0, 1.01), handlelength=1.6)
    panel_letter(ax_a, "a", "security predicate vs. adversary strength")

    # (b) AUC
    ax_b.axhspan(0.85, 0.90, color=C_ATT, alpha=0.07, lw=0)
    rule(ax_b, 0.90, "0.90 floor", dy=0)
    ax_b.errorbar(x, auc, yerr=[lo, hi], fmt="none", ecolor=C_DET, elinewidth=1.0, capsize=2.5, capthick=0.8, zorder=3)
    ax_b.plot(x, auc, "-", color=C_DET, lw=1.6, zorder=3)
    ax_b.plot(x, auc, "o", color=C_DET, ms=5, **mk)
    last = p7[-1]
    ax_b.annotate(f"{auc[-1]:.3f}\nCI [{last['auc_ci95'][0]:.3f}, {last['auc_ci95'][1]:.3f}]",
                  (x[-1], auc[-1]), xytext=(6, 4), textcoords="offset points", fontsize=FS_SMALL,
                  color=INK2, va="bottom", ha="left")
    ax_b.set_ylim(0.86, 1.012); ax_b.set_xlim(*xlim)
    ax_b.set_yticks([0.88, 0.92, 0.96, 1.0]); ax_b.set_yticklabels(["0.88", "0.92", "0.96", "1.00"])
    ax_b.set_ylabel("AUC (sealed)")
    panel_letter(ax_b, "b", "discrimination, 95% BCa CI")

    # (c) decomposition
    # same encoding as panel (a): generation-time = dashed + hollow square, sealed = solid + filled circle
    ax_c.plot(x, emitv, "--", color=C_DET, lw=1.3, label="tool-call emission (gen-time)", zorder=3)
    ax_c.plot(x, emitv, "s", color=C_DET, ms=4.5, mfc=SURFACE, mec=C_DET, mew=1.0, zorder=5)
    ax_c.plot(x, evade, "-", color=C_ATT, lw=1.8, label="gate evasion (sealed)", zorder=3)
    ax_c.plot(x, evade, "o", color=C_ATT, ms=5, **mk)
    ax_c.annotate(f"emits {emitv[-1]:.2f}", (x[-1], emitv[-1]), xytext=(5, -3), textcoords="offset points",
                  fontsize=FS_SMALL, color=INK2, va="top")
    ax_c.annotate(f"evades {evade[-1]:.2f}", (x[-1], evade[-1]), xytext=(5, 2), textcoords="offset points",
                  fontsize=FS_SMALL, color=INK2, va="bottom")
    ax_c.set_ylim(-0.04, 1.32); ax_c.set_xlim(*xlim)
    ax_c.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_c.set_ylabel("rate")
    ax_c.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), handlelength=1.6)
    panel_letter(ax_c, "c", "the step is evasion, not more emission")

    ax_c.set_xticks(x); ax_c.set_xticklabels([f"{l:g}" for l in lam])
    ax_c.set_xlabel(r"detector-awareness $\lambda_{\mathrm{det}}$")
    save(fig, "fig1_boundary")


# ------------------------------------------------------------------- fig3: FPR by SNR
def fig3():
    b = _load("out/phase7/sealed_result.json")["exploratory_benign_decomposition"]["confirmatory_set_benign_fpr"]
    snr = b["by"]["snr_bucket"]
    order = ["high", "mid", "low"]
    names = {"high": "high SNR", "mid": "mid SNR", "low": "low SNR"}
    n = np.array([snr[k]["n"] for k in order]); k = np.array([snr[k]["k"] for k in order])
    fpr = [snr[q]["fpr"] for q in order]
    N, K = int(b["n"]), int(b["k"])
    assert n.sum() == N and k.sum() == K, "SNR buckets must partition the confirmatory set"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 2.2), gridspec_kw={"width_ratios": [1.6, 1.0], "wspace": 0.5})
    x = np.arange(3) * 1.15
    ax1.bar(x, fpr, width=0.62, color=BLUES, zorder=3)
    ax1.axhline(0.02, color=RULE, lw=0.7, zorder=1)
    xg = (x[1] + x[2]) / 2   # centred in the gap between the mid and low bars; two lines so it fits the gap
    ax1.annotate("gate\n0.02", xy=(xg, 0.02), xytext=(xg, 0.14), ha="center", va="bottom", fontsize=6.6, color=INK2,
                 linespacing=1.0, arrowprops=dict(arrowstyle="-", lw=0.5, color=RULE, shrinkA=0, shrinkB=0))   # short leader to the line
    for xi, f, ni in zip(x, fpr, n):
        ax1.annotate(f"{f:.3f}", (xi, f), xytext=(0, 3), textcoords="offset points", ha="center",
                     va="bottom", fontsize=FS_SMALL, color=INK)
    ax1.set_xticks(x); ax1.set_xticklabels([f"{q}\n$n$={ni:,}" for q, ni in zip(order, n)], fontsize=6.6)
    ax1.set_xlabel("SNR bucket", fontsize=7.5)
    ax1.set_ylim(0, 0.47); ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax1.set_ylabel("benign false-positive rate")
    ax1.text(0.0, 1.03, "(a)", transform=ax1.transAxes, fontsize=8, fontweight="bold", va="bottom")
    ax1.text(0.40, 0.97, f"overall {b['fpr']:.4f}\n({K}/{N:,})", transform=ax1.transAxes, fontsize=FS_SMALL,
             va="top", ha="center", color=INK2)

    # (b) composition: share of clips vs share of false positives, same ramp, 2px surface gaps
    clip_share = n / N; fp_share = k / K
    cols = ["clips", "false\npositives"]
    bottoms = np.zeros(2)
    for i, q in enumerate(order):
        vals = np.array([clip_share[i], fp_share[i]])
        ax2.bar([0, 1], vals, bottom=bottoms, width=0.58, color=BLUES[i], edgecolor=SURFACE, linewidth=1.2, zorder=3)
        bottoms += vals
    noisy_clip = clip_share[1] + clip_share[2]; noisy_fp = fp_share[1] + fp_share[2]
    ax2.annotate(f"noisy\n{noisy_clip*100:.1f}%", (0, 1.0), xytext=(0, 3), textcoords="offset points",
                 ha="center", va="bottom", fontsize=FS_SMALL, color=INK)
    ax2.annotate(f"noisy\n{noisy_fp*100:.0f}%", (1, 1.0), xytext=(0, 3), textcoords="offset points",
                 ha="center", va="bottom", fontsize=FS_SMALL, color=INK)
    ax2.text(0, clip_share[0] / 2, f"{clip_share[0]*100:.0f}%", ha="center", va="center", fontsize=FS_SMALL, color=INK)
    ax2.text(1, fp_share[0] / 2, f"{fp_share[0]*100:.0f}%", ha="center", va="center", fontsize=FS_SMALL, color=INK)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(cols, fontsize=FS_SMALL)
    ax2.set_ylim(0, 1.22); ax2.set_yticks([0, 0.5, 1.0]); ax2.set_yticklabels(["0%", "50%", "100%"])
    # no y-label: the percentage ticks and the caption say "share"; the gutter is too narrow for one
    ax2.legend(handles=[Patch(color=BLUES[i], label=names[q]) for i, q in enumerate(order)],
               loc="upper left", bbox_to_anchor=(1.02, 1.0), handlelength=1.0, handleheight=1.0, fontsize=FS_SMALL)
    ax2.text(0.0, 1.03, "(b)", transform=ax2.transAxes, fontsize=8, fontweight="bold", va="bottom")
    save(fig, "fig2_fpr_by_snr")


# ---------------------------------------------------------------- fig4: channel robustness
def fig4():
    chans = [("clean", "clean"), ("codec_opus", "Opus\n16 kbps"), ("codec_vorbis", "Vorbis\n64 kbps"),
             ("ota_20", "OTA\n20 dB"), ("ota_10", "OTA\n10 dB")]
    pos = np.array([0, 1, 2, 3.4, 4.4])
    fpr, auc1, labels = [], [], []
    for key, lab in chans:
        d = _load(f"out/phase8/channel_{key}.json")
        fpr.append(d["benign_fpr"]["overall"]["fpr"]); auc1.append(d["detector_auc"]["1.0"]["auc"]); labels.append(lab)
    # SALMONN actuation on the lambda=1 dev-seed triggers where it was measured: the clean channel
    # is the emission-null control (X1), codec/OTA come from the Phase-8 actuation runs.
    act = {}
    act["clean"] = _load("out/phase8/actuation_nulls_lam1.json")["strict_actuation"]["trigger"]
    for key in ("codec_opus", "ota_20"):
        act[key] = _load(f"out/phase8/act/channel_{key}.json")["actuation"]["1.0"]["actuation_rate"]

    fig, axes = plt.subplots(3, 1, figsize=(W, 4.4), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.5})
    ax_a, ax_b, ax_c = axes
    w = 0.6
    for ax in axes:
        ax.axvline(2.7, color=RULE, lw=0.6, ls=(0, (2, 2)), zorder=1)
        ax.set_xlim(-0.6, 5.0)
    ax_c.text(1.5, -0.62, "codec transcoding", transform=ax_c.get_xaxis_transform(), ha="center", va="top", fontsize=FS_SMALL, color=INK2)
    ax_c.text(3.9, -0.62, "over-the-air", transform=ax_c.get_xaxis_transform(), ha="center", va="top", fontsize=FS_SMALL, color=INK2)

    # (a) AUC — three decimals so 0.999 is not read as a perfect 1.00
    ax_a.axhspan(0.0, 0.90, color=C_ATT, alpha=0.05, lw=0)
    rule(ax_a, 0.90, "0.90 floor", dy=0)
    ax_a.bar(pos, auc1, width=w, color=C_DET, zorder=3)
    for p, v in zip(pos, auc1):
        ax_a.annotate(f"{v:.3f}", (p, v), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=FS_SMALL, color=INK)
    ax_a.set_ylim(0, 1.14); ax_a.set_yticks([0, 0.5, 1.0]); ax_a.set_ylabel(r"AUC ($\lambda_{\mathrm{det}}$=1)")
    panel_letter(ax_a, "a", "detector discrimination")

    # (b) benign FPR — three decimals: the codec bars sit just above the 0.02 gate
    rule(ax_b, 0.02, "gate 0.02", dy=0)
    ax_b.bar(pos, fpr, width=w, color=C_DET, zorder=3)
    for p, v in zip(pos, fpr):
        ax_b.annotate(f"{v:.3f}", (p, v), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=FS_SMALL, color=INK)
    ax_b.set_ylim(0, 0.92); ax_b.set_yticks([0, 0.4, 0.8]); ax_b.set_ylabel("benign FPR")
    panel_letter(ax_b, "b", "false positives on clean-trained detector")

    # (c) actuation (measured channels only)
    for key, lab in chans:
        p = pos[labels.index(lab)]
        if key in act:
            v = act[key]
            ax_c.bar([p], [v], width=w, color=C_ATT, zorder=3)
            ax_c.annotate(f"{v:.2f}", (p, v), xytext=(0, 2), textcoords="offset points",
                          ha="center", va="bottom", fontsize=FS_SMALL, color=INK)
        else:
            ax_c.text(p, 0.04, "not\nmeasured", ha="center", va="bottom", fontsize=FS_SMALL, color=INK2, style="italic")
    ax_c.set_ylim(0, 1.05); ax_c.set_yticks([0, 0.5, 1.0]); ax_c.set_ylabel("SALMONN actuation")
    panel_letter(ax_c, "c", r"attack survival ($\lambda_{\mathrm{det}}$=1 triggers)")
    ax_c.set_xticks(pos); ax_c.set_xticklabels(labels, fontsize=FS_SMALL)
    save(fig, "fig3_channel")


if __name__ == "__main__":
    for stale in ("fig3_fpr_by_snr", "fig4_channel", "fig2_auc"):   # pre-redesign names
        for ext in ("pdf", "png"):
            (OUT / f"{stale}.{ext}").unlink(missing_ok=True)
    fig1(); fig3(); fig4()
    print("all figures written to", OUT)
