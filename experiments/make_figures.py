#!/usr/bin/env python3
"""Publication figures from the experiment results (vector PDF + PNG preview).

  python experiments/make_figures.py --results results --fig-eps 3

Figures (results/figures/):
  fig_privacy_utility.pdf   RF AUPRC vs privacy budget, common vs combined, real-data reference
  fig_baselines.pdf         RF AUPRC vs privacy budget, ours vs every baseline
  fig_gains.pdf             combined - common difference per seed and its 95% CI, per dataset and budget
  fig_ablation.pdf          ablation of the sampler (needs --ablations runs)
  fig_pr_curves.pdf         test precision-recall curves with AP: real data, common, combined (seed 0)
  fig_mia_roc_domias.pdf    DOMIAS-style attack ROC on log-FPR axes with AUC, common vs combined (seed 0)
  fig_sampler_scores.pdf    flow likelihood-ratio scores: pool, before/after refinement, selected
  fig_marginals_<ds>.pdf    minority-class marginals with KS: real vs common vs rare stream
  fig_embedding_<ds>.pdf    t-SNE of real minority, common-stream minority and rare stream
  fig_embedding_all.pdf     the four t-SNE panels side by side
  fig_corrdiff_<ds>.pdf     |corr(real) - corr(synthetic)| (lower triangle) for common and combined

Style: serif text close to the ACM template font (Linux Libertine if installed, else STIX),
the Okabe-Ito colour-blind-safe palette, and a distinct marker / line style for every series so
the figures stay readable in greyscale print. Figures are drawn at their printed size
(6.8 in = ACM text width; 3.3 in = one column), so include them with width=\\textwidth or
\\columnwidth and the fonts print at 6.5-8 pt. Error bands and bars are 95% t-intervals over seeds.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from common import DATASET_LABELS, DATASETS, load, make_splits

# ---- style -------------------------------------------------------------------------------------
# Okabe-Ito palette; neutral greys for references.
COMBINED, COMMON, REAL = "#0072B2", "#D55E00", "#4d4d4d"
INK, MUTED, LIGHT, GRID = "#1a1a1a", "#6b6b6b", "#b8b8b8", "#ebebeb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f7fbff", "#d6e6f4", "#abd0e6", "#6aaed6", "#3787c0",
                                                     "#105ba4", "#08306b"])
STYLE = {"common": dict(color=COMMON, marker="o", ls="--", label="Common only", short="Common"),
         "combined": dict(color=COMBINED, marker="s", ls="-", label="Combined (with rare stream)",
                          short="Combined"),
         "trtr": dict(color=REAL, ls=":", label="Real data (TRTR)", short="Real")}
BASELINE_STYLE = {"pategan": dict(color="#009E73", marker="^", ls="-.", label="PATE-GAN"),
                  "dpgan": dict(color="#CC79A7", marker="v", ls=(0, (4, 1.5, 1, 1.5)), label="DP-GAN"),
                  "aim": dict(color="#E69F00", marker="D", ls=(0, (1, 1.2)), label="AIM", ms=3.2),
                  "ctgan": dict(color="#56B4E9", marker="P", ls="none", label="CTGAN (no DP)", ms=4.4),
                  "tvae": dict(color=INK, marker="X", ls="none", label="TVAE (no DP)", ms=4.2),
                  "smote": dict(color=MUTED, marker="*", ls="none", label="SMOTE (no DP)", ms=5.2),
                  "sn_dpctgan": dict(color="#E69F00", marker="D", ls=(0, (1, 1.2)), label="DP-CTGAN (SmartNoise)",
                                     ms=3.2),
                  "sn_patectgan": dict(color="#F0E442", marker="h", ls=(0, (3, 1)), label="PATE-CTGAN (SmartNoise)"),
                  "sn_mst": dict(color="#882255", marker="p", ls=(0, (5, 1, 1, 1)), label="MST (SmartNoise)")}
EPS_POS = ["1", "3", "10", "nodp"]
EPS_LBL = {"1": "1", "3": "3", "10": "10", "nodp": "No DP"}
FULL_W, HALF_W = 6.8, 3.3   # inches (ACM text width ~ 6.8 in, column ~ 3.3 in)
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _font():
    have = {f.name for f in font_manager.fontManager.ttflist if f.fname.lower().endswith(".ttf")}  # .otf/CFF embeds badly
    for name in ("Linux Libertine O", "Libertinus Serif", "Linux Libertine", "STIX Two Text", "STIXGeneral"):
        if name in have:
            return name
    return "DejaVu Serif"


FONT = _font()
_math = ({"mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
          "mathtext.bf": f"{FONT}:bold", "mathtext.fallback": "stix"}
         if "ibertin" in FONT else {"mathtext.fontset": "stix"})
plt.rcParams.update({
    "font.family": "serif", "font.serif": [FONT, "STIXGeneral", "DejaVu Serif"], **_math,
    "font.size": 7.5, "axes.titlesize": 8, "axes.titlelocation": "left", "axes.titlepad": 4,
    "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 7,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "axes.linewidth": 0.6,
    "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "xtick.minor.width": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.25, "lines.markersize": 3.8, "lines.markeredgewidth": 0.5,
    "legend.frameon": False, "legend.handlelength": 2.4, "legend.columnspacing": 1.5,
    "legend.handletextpad": 0.5, "legend.borderaxespad": 0.2,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"), metadata={"Creator": None, "Producer": None,
                                                            "CreationDate": None})
    fig.savefig(os.path.join(out, name + ".png"), dpi=200)
    plt.close(fig)
    print("  wrote", name)


def ci95(x):
    x = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce").dropna())
    if len(x) < 2:
        return (x.mean() if len(x) else np.nan), 0.0
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def panel_title(ax, i, text):
    ax.set_title(f"({LETTERS[i]}) {text}")


def top_legend(fig, axes, ncol, handles=None, labels=None, y=1.0):
    """One legend above the whole figure, built from the labelled artists of all panels."""
    if handles is None:
        seen = {}
        for ax in np.ravel(axes):
            for h, lab in zip(*ax.get_legend_handles_labels()):
                if not lab.startswith("_"):
                    seen.setdefault(lab, h)
        handles, labels = list(seen.values()), list(seen)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol)


def note(ax, lines, loc="lower left", y0=None):
    """Coloured value annotations, e.g. [("AP 0.74", colour), ...], stacked in a panel corner
    (y0: axes-fraction height of the first line instead of the corner)."""
    x, ha = (0.04, "left") if "left" in loc else (0.96, "right")
    top = "upper" in loc
    for k, (txt, c) in enumerate(lines):
        if y0 is not None:
            y = y0 - 0.1 * k
        else:
            y = (0.95 - 0.1 * k) if top else (0.05 + 0.1 * (len(lines) - 1 - k))
        ax.text(x, y, txt, color=c, fontsize=6.3, ha=ha, va="top" if top else "bottom", transform=ax.transAxes)


# ---- privacy-budget axis helpers ---------------------------------------------------------------
def _xs(pos):
    """x positions: DP budgets evenly spaced, No-DP set apart by an extra half step."""
    return np.array([i + (0.5 if e == "nodp" else 0.0) for i, e in enumerate(pos)])


def _hist_key(ax, color, ls, label):
    """Legend entry drawn as a line (step histograms would otherwise show as boxes)."""
    ax.plot([], [], color=color, ls=ls, lw=1.2, label=label)


def _eps_axis(ax, pos):
    xs = _xs(pos)
    ax.set_xticks(xs, [EPS_LBL[e] for e in pos])
    ax.set_xlim(xs[0] - 0.45, xs[-1] + 0.45)
    if pos[-1] == "nodp" and len(pos) > 1:
        ax.axvline((xs[-1] + xs[-2]) / 2, color=LIGHT, lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax.set_xlabel(r"Privacy budget $\varepsilon$")
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def _eps_series(ax, pos, groups, st, dx=0.0, band=True, zorder=3):
    """Mean over seeds per budget. DP budgets are joined by a line with a 95% CI band (or bars);
    the No-DP point stands alone, because it is not part of the epsilon scale."""
    xs = _xs(pos) + dx
    ms_ = [ci95(g) for g in groups]
    m, h = np.array([v[0] for v in ms_], float), np.array([v[1] for v in ms_], float)
    # metrics are in [0, 1]: draw intervals clipped to that range (asymmetric bars)
    lo, hi = np.clip(m - h, 0, 1), np.clip(m + h, 0, 1)
    h_lo, h_hi = np.nan_to_num(m - lo), np.nan_to_num(hi - m)
    ok = ~np.isnan(m)
    dp, nd = ok & np.array([e != "nodp" for e in pos]), ok & np.array([e == "nodp" for e in pos])
    mk = dict(color=st["color"], marker=st["marker"], ms=st.get("ms", 3.8), mec="white", mew=0.5)
    if dp.any():
        ax.plot(xs[dp], m[dp], ls=st.get("ls", "-"), label=st["label"], zorder=zorder, **mk)
        if band and dp.sum() > 1:
            ax.fill_between(xs[dp], lo[dp], hi[dp], color=st["color"], alpha=0.14, lw=0, zorder=zorder - 1)
        elif h[dp].any():
            ax.errorbar(xs[dp], m[dp], yerr=[h_lo[dp], h_hi[dp]], fmt="none", ecolor=st["color"], elinewidth=0.8,
                        capsize=1.8, zorder=zorder)
    if nd.any():
        ax.errorbar(xs[nd], m[nd], yerr=[h_lo[nd], h_hi[nd]], ls="none", elinewidth=0.8, capsize=1.8, zorder=zorder,
                    label=None if dp.any() else st["label"], **mk)


# ---- figures from runs.csv -----------------------------------------------------------------
def fig_privacy_utility(df, out, metric="rf_auprc", ylabel="RF AUPRC (test)"):
    ds_list = [d for d in DATASETS if d in df.dataset.unique()]
    fig, axes = plt.subplots(1, len(ds_list), figsize=(FULL_W, 2.0), squeeze=False, layout="constrained")
    for i, (ax, ds) in enumerate(zip(axes[0], ds_list)):
        sub = df[(df.method == "ours") & (df.dataset == ds)]
        pos = [e for e in EPS_POS if e in sub.eps.unique()]
        for cfg, dx in (("common", -0.05), ("combined", 0.05)):
            _eps_series(ax, pos, [sub[(sub.eps == e) & (sub.config == cfg)][metric] for e in pos], STYLE[cfg], dx=dx)
        ax.axhline(sub[sub.config == "trtr"][metric].mean(), color=REAL, ls=":", lw=1.1, label=STYLE["trtr"]["label"],
                   zorder=2)
        _eps_axis(ax, pos)
        panel_title(ax, i, DATASET_LABELS[ds])
    axes[0][0].set_ylabel(ylabel)
    top_legend(fig, axes, ncol=3)
    save(fig, out, "fig_privacy_utility")


def fig_baselines(df, out, metric="rf_auprc", ylabel="RF AUPRC (test)"):
    meths = [m for m in BASELINE_STYLE if (df.method == m).any()]
    if not meths:
        print("  (no baseline runs found; skipping fig_baselines)")
        return
    ds_list = [d for d in DATASETS if d in df[df.method == "ours"].dataset.unique()]
    fig, axes = plt.subplots(1, len(ds_list), figsize=(FULL_W, 2.1), squeeze=False, layout="constrained")
    offs = dict(zip(meths, np.linspace(-0.24, 0.24, len(meths)) if len(meths) > 1 else [0.12]))
    ours = dict(STYLE["combined"], label="Ours (combined)", ms=4.2)
    for i, (ax, ds) in enumerate(zip(axes[0], ds_list)):
        sub = df[df.dataset == ds]
        pos = [e for e in EPS_POS if e in sub[sub.method == "ours"].eps.unique()]
        for meth in meths:
            s = sub[sub.method == meth]
            _eps_series(ax, pos, [s[s.eps == e][metric] for e in pos], BASELINE_STYLE[meth], dx=offs[meth], band=False,
                        zorder=3)
        s = sub[(sub.method == "ours") & (sub.config == "combined")]
        _eps_series(ax, pos, [s[s.eps == e][metric] for e in pos], ours, zorder=5)
        ax.axhline(sub[(sub.method == "ours") & (sub.config == "trtr")][metric].mean(), color=REAL, ls=":", lw=1.1,
                   label=STYLE["trtr"]["label"], zorder=2)
        _eps_axis(ax, pos)
        panel_title(ax, i, DATASET_LABELS[ds])
    axes[0][0].set_ylabel(ylabel)
    seen = {}
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    order = [ours["label"]] + [BASELINE_STYLE[m]["label"] for m in meths] + [STYLE["trtr"]["label"]]
    order = [o for o in order if o in seen]
    top_legend(fig, axes, ncol=(len(order) + 1) // 2, handles=[seen[o] for o in order], labels=order)
    save(fig, out, "fig_baselines")


def fig_gains(df, out, metrics=(("rf_auprc", "RF AUPRC"), ("knn_auprc", "$k$-NN AUPRC"), ("rf_auc", "RF AUC"))):
    ours = df[df.method == "ours"]
    a = ours[ours.config == "combined"].set_index(["dataset", "eps", "seed"])
    b = ours[ours.config == "common"].set_index(["dataset", "eps", "seed"])
    idx = a.index.intersection(b.index)
    have = {(i[0], i[1]) for i in idx}
    ds_list = [d for d in DATASETS if any((d, e) in have for e in EPS_POS)]
    fig, axes = plt.subplots(len(ds_list), len(metrics), figsize=(FULL_W, 0.68 * len(ds_list) + 0.55),
                             sharex="col", squeeze=False, layout="constrained")
    for j, (m, title) in enumerate(metrics):
        d = pd.to_numeric(a.loc[idx, m]) - pd.to_numeric(b.loc[idx, m])
        for i, ds in enumerate(ds_list):
            ax = axes[i][j]
            pos = [e for e in EPS_POS if (ds, e) in have]
            for k, e in enumerate(pos):
                v = d.xs((ds, e), level=("dataset", "eps")).dropna()
                mean, h = ci95(v)
                sig = mean - h > 0 or mean + h < 0
                ax.scatter(v.values, np.full(len(v), k), s=5, color=COMBINED, alpha=0.35, lw=0, zorder=2)
                ax.errorbar(mean, k, xerr=h, fmt="o", color=COMBINED, mfc=COMBINED if sig else "white", mec=COMBINED,
                            ms=4, mew=0.9, elinewidth=1.0, capsize=2, zorder=3)
            ax.axvline(0, color=MUTED, lw=0.7, zorder=1)
            ax.set_yticks(range(len(pos)), [EPS_LBL[e] for e in pos])
            ax.set_ylim(len(pos) - 0.5, -0.5)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x", visible=True)
            ax.grid(axis="y", visible=False)
            if j == 0:
                ax.set_ylabel(DATASET_LABELS[ds], rotation=0, ha="right", va="center")
            if i == 0:
                panel_title(ax, j, title)
            if i == len(ds_list) - 1:
                ax.set_xlabel(r"Combined $-$ common")
    mk = dict(marker="o", ls="none", color=COMBINED, mec=COMBINED, ms=4, mew=0.9)
    top_legend(fig, axes, ncol=3, handles=[Line2D([], [], mfc=COMBINED, **mk), Line2D([], [], mfc="white", **mk),
                                           Line2D([], [], marker="o", ls="none", color=COMBINED, alpha=0.35, ms=2.5,
                                                  mew=0)],
               labels=["Mean, 95% CI excludes 0", "Mean, 95% CI includes 0", "Individual seeds"])
    save(fig, out, "fig_gains")


def fig_ablation(df, out, eps="3", metric="rf_auprc"):
    sub = df[(df.method == "ours") & (df.eps == eps)]
    variants = [("common", "Common only"), ("abl:plain_oversample", "+ plain minority samples"),
                ("abl:no_flow_refine", "w/o flow + refinement"), ("abl:no_refine", "w/o refinement"),
                ("abl:no_judge", "w/o judge"), ("abl:no_maxmin", "w/o diversity (MaxMin)"),
                ("abl:no_qm", "w/o quantile matching"),
                ("abl:judge_disc", "judge = DP discriminator"),
                ("abl:judge_rf_nonprivate", "judge = non-private RF"), ("combined", "Full sampler")]
    variants = [v for v in variants if (sub.config == v[0]).any()]
    if len(variants) < 3:
        print("  (no ablation runs found; skipping fig_ablation)")
        return
    ds_list = [d for d in DATASETS if d in sub.dataset.unique()]
    fig, axes = plt.subplots(1, len(ds_list), figsize=(FULL_W, 0.2 * len(variants) + 0.95), sharey=True,
                             squeeze=False, layout="constrained")
    for p, (ax, ds) in enumerate(zip(axes[0], ds_list)):
        s = sub[sub.dataset == ds]
        full_m, full_h = ci95(s[s.config == "combined"][metric])
        ax.axvspan(full_m - full_h, full_m + full_h, color=COMBINED, alpha=0.1, lw=0, zorder=0)
        ax.axvline(full_m, color=COMBINED, lw=0.7, ls="-", zorder=1)
        ax.axvline(s[s.config == "common"][metric].mean(), color=COMMON, lw=0.7, ls="--", zorder=1)
        for i, (cfg, _) in enumerate(variants):
            v = pd.to_numeric(s[s.config == cfg][metric], errors="coerce").dropna()
            mean, h = ci95(v)
            c, mk = (COMBINED, "s") if cfg == "combined" else ((COMMON, "o") if cfg == "common" else (INK, "D"))
            ax.scatter(v.values, np.full(len(v), i), s=4, color=c, alpha=0.3, lw=0, zorder=2)
            ax.errorbar(mean, i, xerr=h, fmt=mk, color=c, mec="white", mew=0.5, ms=4 if mk != "D" else 3.4,
                        elinewidth=0.9, capsize=1.8, zorder=3)
        panel_title(ax, p, DATASET_LABELS[ds])
        ax.set_xlabel(f"RF AUPRC ($\\varepsilon$ = {EPS_LBL.get(eps, eps)})")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0][0].set_yticks(range(len(variants)), [v[1] for v in variants])
    axes[0][0].set_ylim(len(variants) - 0.5, -0.5)
    save(fig, out, "fig_ablation")


# ---- figures from seed artefacts ---------------------------------------------------------------
def art(results, ds, eps, seed, name):
    p = os.path.join(results, ds, f"eps{eps}", f"seed{seed}", name)
    return p if os.path.exists(p) else None


def fig_pr_curves(results, out, eps, seed):
    ds_list = [d for d in DATASETS if art(results, d, eps, seed, "probs.npz")]
    if not ds_list:
        return
    fig, axes = plt.subplots(1, len(ds_list), figsize=(FULL_W, 2.0), squeeze=False, sharey=True,
                             layout="constrained")
    for i, (ax, ds) in enumerate(zip(axes[0], ds_list)):
        z = np.load(art(results, ds, eps, seed, "probs.npz"))
        y = z["y_test"]
        aps = []
        for key, cfg in (("trtr_rf", "trtr"), ("common_rf", "common"), ("combined_rf", "combined")):
            p, r, _ = precision_recall_curve(y, z[key])
            st = STYLE[cfg]
            ax.plot(r, p, color=st["color"], ls=st["ls"], lw=1.2, drawstyle="steps-post", label=st["label"],
                    zorder=3 if cfg == "combined" else 2)
            aps.append((f"{st['short']}  AP {average_precision_score(y, z[key]):.3f}", st["color"]))
        ax.axhline(y.mean(), color=LIGHT, lw=0.8, ls=(0, (3, 2)), zorder=1, label="Prevalence")
        note(ax, aps, "lower left", y0=0.62)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.grid(axis="x", visible=True)
        panel_title(ax, i, DATASET_LABELS[ds])
        ax.set_xlabel("Recall")
    axes[0][0].set_ylabel("Precision")
    top_legend(fig, axes, ncol=4)
    save(fig, out, "fig_pr_curves")


def fig_mia_roc(results, out, eps, seed, attack="domias"):
    ds_list = [d for d in DATASETS if art(results, d, eps, seed, "mia_scores.npz")]
    if not ds_list:
        return
    fig, axes = plt.subplots(1, len(ds_list), figsize=(FULL_W, 2.0), squeeze=False, sharey=True,
                             layout="constrained")
    for i, (ax, ds) in enumerate(zip(axes[0], ds_list)):
        z = np.load(art(results, ds, eps, seed, "mia_scores.npz"))
        g = np.logspace(-3, 0, 100)
        ax.plot(g, g, color=MUTED, lw=0.8, ls=":", label="Random guess", zorder=1)
        aucs = []
        for cfg in ("common", "combined"):
            yt, sc = z[f"{cfg}_y"], z[f"{cfg}_{attack}"]
            fpr, tpr, _ = roc_curve(yt, sc)
            st = STYLE[cfg]
            ax.plot(np.clip(fpr, 1e-3, 1), tpr, color=st["color"], ls=st["ls"], lw=1.2, drawstyle="steps-post",
                    label=st["label"], zorder=3 if cfg == "combined" else 2)
            aucs.append((f"{st['short']}  AUC {roc_auc_score(yt, sc):.3f}", st["color"]))
        note(ax, aucs, "upper left")
        ax.set_xscale("log"); ax.set_xlim(1e-3, 1); ax.set_ylim(0, 1.02)
        ax.grid(axis="x", which="major", visible=True)
        panel_title(ax, i, DATASET_LABELS[ds])
        ax.set_xlabel("False-positive rate (log scale)")
    axes[0][0].set_ylabel("True-positive rate")
    top_legend(fig, axes, ncol=3)
    save(fig, out, f"fig_mia_roc_{attack}")


def fig_sampler_scores(results, out, eps, seed):
    ds_list = [d for d in DATASETS if art(results, d, eps, seed, "sampler_diag.npz")]
    if not ds_list:
        return
    fig, axes = plt.subplots(1, len(ds_list), figsize=(FULL_W, 1.85), squeeze=False, layout="constrained")
    series = [("pool_score", "Stage-1 pool", "fill"), ("refined_before", "Before refinement", COMMON, "--"),
              ("refined_after", "After refinement", COMBINED, "-"), ("selected_score", "Selected", INK, ":")]
    for i, (ax, ds) in enumerate(zip(axes[0], ds_list)):
        z = np.load(art(results, ds, eps, seed, "sampler_diag.npz"))
        allv = np.concatenate([z[s[0]] for s in series if s[0] in z and len(z[s[0]])])
        lo, hi = np.percentile(allv, [0.5, 99.5])
        bins = np.linspace(lo, hi, 50)
        for k, lab, *sty in series:
            if k not in z or not len(z[k]):
                continue
            v = z[k][(z[k] >= lo) & (z[k] <= hi)]
            if sty[0] == "fill":
                ax.hist(v, bins=bins, density=True, histtype="stepfilled", color=LIGHT, alpha=0.55, lw=0, label=lab)
            else:
                ax.hist(v, bins=bins, density=True, histtype="step", color=sty[0], ls=sty[1], lw=1.2)
                _hist_key(ax, sty[0], sty[1], lab)
        panel_title(ax, i, DATASET_LABELS[ds])
        ax.set_xlabel(r"$\log p_{\mathrm{min}}(h) - \log p_{\mathrm{maj}}(h)$")
        ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.grid(False)
        ax.xaxis.set_major_locator(MaxNLocator(4))
    axes[0][0].set_ylabel("Density")
    top_legend(fig, axes, ncol=4)
    save(fig, out, "fig_sampler_scores")


def _real_and_synth(results, ds, eps, seed, data_dir):
    pc, pr = art(results, ds, eps, seed, "synthetic_common.csv.gz"), art(results, ds, eps, seed, "synthetic_rare.csv.gz")
    if not (pc and pr):
        return None
    cfg, df, target, cat_cols = load(ds, data_dir)
    train = make_splits(df, target, seed)["train"]
    return train, pd.read_csv(pc), pd.read_csv(pr), target, cat_cols


def fig_marginals(results, out, ds, eps, seed, data_dir, k=4):
    got = _real_and_synth(results, ds, eps, seed, data_dir)
    if got is None:
        return
    train, common, rare, target, cat_cols = got
    cont = [c for c in train.columns if c not in cat_cols + [target] and train[c].nunique() > 10]
    rmin, rmaj = train[train[target] == 1], train[train[target] == 0]
    sep = sorted(cont, key=lambda c: -stats.ks_2samp(rmin[c], rmaj[c]).statistic)[:k]
    if not sep:
        return
    fig, axes = plt.subplots(1, len(sep), figsize=(FULL_W, 1.85), squeeze=False, layout="constrained")
    cmin = common[common[target] == 1]
    for i, (ax, c) in enumerate(zip(axes[0], sep)):
        lo, hi = np.percentile(np.r_[rmin[c], cmin[c], rare[c]], [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        inr = lambda v: v[(v >= lo) & (v <= hi)]
        ax.hist(inr(rmin[c]), bins=bins, density=True, histtype="stepfilled", color=LIGHT, alpha=0.6, lw=0,
                label="Real minority (training)")
        ks = []
        for frame, st, lab in ((cmin, STYLE["common"], "Common-stream minority"),
                               (rare, STYLE["combined"], "Rare stream")):
            if len(frame):
                ax.hist(inr(frame[c]), bins=bins, density=True, histtype="step", color=st["color"], ls=st["ls"],
                        lw=1.2)
                _hist_key(ax, st["color"], st["ls"], lab)
                ks.append(f"{stats.ks_2samp(rmin[c], frame[c]).statistic:.2f}")
            else:
                ks.append("--")
        panel_title(ax, i, str(c))
        ax.set_xlabel(f"KS: common {ks[0]}, rare {ks[1]}", fontsize=6.5, color=MUTED)
        ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.grid(False)
        ax.xaxis.set_major_locator(MaxNLocator(4))
    top_legend(fig, axes, ncol=3)
    save(fig, out, f"fig_marginals_{ds}")


def _embedding(results, ds, eps, seed, data_dir, n=1000):
    got = _real_and_synth(results, ds, eps, seed, data_dir)
    if got is None:
        return None
    train, common, rare, target, _ = got
    feat = [c for c in train.columns if c != target]
    rng = np.random.RandomState(0)
    take = lambda d: d.iloc[rng.choice(len(d), min(n, len(d)), replace=False)][feat].values.astype(float)
    groups = [("Real minority (training)", take(train[train[target] == 1]), REAL, "o", 0.45),
              ("Common-stream minority", take(common[common[target] == 1]), COMMON, "^", 0.55),
              ("Rare stream", take(rare), COMBINED, "s", 0.55)]
    groups = [g for g in groups if len(g[1])]
    X = StandardScaler().fit(groups[0][1]).transform(np.vstack([g[1] for g in groups]))
    Z = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(X)
    return Z, [(lab, len(G), c, mk, al) for lab, G, c, mk, al in groups]


def _scatter_embedding(ax, Z, groups):
    i = 0
    for lab, n, c, mk, al in groups:
        ax.scatter(Z[i:i + n, 0], Z[i:i + n, 1], s=4, color=c, marker=mk, alpha=al, lw=0, label=lab, rasterized=True)
        i += n
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_aspect("equal", adjustable="datalim")
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(LIGHT); s.set_linewidth(0.6)


def _legend_handles(groups):
    return ([Line2D([], [], marker=mk, ls="none", color=c, ms=4, mew=0) for _, _, c, mk, _ in groups],
            [g[0] for g in groups])


def fig_embeddings(results, out, eps, seed, data_dir):
    emb = {}
    for ds in DATASETS:
        e = _embedding(results, ds, eps, seed, data_dir)
        if e is None:
            continue
        emb[ds] = e
        fig, ax = plt.subplots(figsize=(HALF_W, HALF_W), layout="constrained")
        _scatter_embedding(ax, *e)
        ax.set_title(f"{DATASET_LABELS[ds]} (t-SNE)")
        h, lab = _legend_handles(e[1])
        ax.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1)
        save(fig, out, f"fig_embedding_{ds}")
    if len(emb) > 1:
        fig, axes = plt.subplots(1, len(emb), figsize=(FULL_W, FULL_W / len(emb) + 0.1), squeeze=False,
                                 layout="constrained")
        for i, (ax, (ds, e)) in enumerate(zip(axes[0], emb.items())):
            _scatter_embedding(ax, *e)
            panel_title(ax, i, DATASET_LABELS[ds])
        h, lab = _legend_handles(max((e[1] for e in emb.values()), key=len))
        top_legend(fig, axes, ncol=3, handles=h, labels=lab)
        save(fig, out, "fig_embedding_all")


def fig_corrdiff(results, out, ds, eps, seed, data_dir, max_labels=24):
    got = _real_and_synth(results, ds, eps, seed, data_dir)
    if got is None:
        return
    train, common, rare, target, _ = got
    feat = [c for c in train.columns if c != target]
    n = len(feat)
    combined = pd.concat([common, rare], ignore_index=True)
    cr = train[feat].corr().fillna(0).values
    mats = [("Common only", np.abs(cr - common[feat].corr().fillna(0).values)),
            ("Combined", np.abs(cr - combined[feat].corr().fillna(0).values))]
    upper = np.triu(np.ones((n, n), bool))
    vmax = max(m[~upper].max() for _, m in mats)
    labels = n <= max_labels
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W * (0.92 if labels else 0.72), FULL_W * (0.46 if labels else 0.36)),
                             layout="constrained")
    for i, (ax, (t, m)) in enumerate(zip(axes, mats)):
        im = ax.imshow(np.ma.masked_where(upper, m), cmap=SEQ, vmin=0, vmax=vmax, interpolation="nearest")
        panel_title(ax, i, f"{t}: mean {m[~upper].mean():.3f}")
        ax.set_xlim(-0.5, n - 1.5); ax.set_ylim(n - 0.5, 0.5)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        if labels:
            ax.set_xticks(range(n - 1), feat[:-1], rotation=90, fontsize=5)
            ax.set_yticks(range(1, n), feat[1:], fontsize=5)
            ax.tick_params(length=0, pad=1.5)
            if i == 1:
                ax.set_yticklabels([])
        else:
            ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=list(axes), fraction=0.03, pad=0.02, shrink=0.85)
    cb.set_label(r"$|\rho_{\mathrm{real}} - \rho_{\mathrm{synth}}|$")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=0.5)
    save(fig, out, f"fig_corrdiff_{ds}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--fig-eps", default="3", help="budget used for seed-level figures (1, 3, 10 or nodp)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--ablation-eps", default="3")
    a = ap.parse_args()
    out = os.path.join(a.results, "figures")
    os.makedirs(out, exist_ok=True)
    print(f"  font: {FONT}")
    runs = os.path.join(a.results, "summary", "runs.csv")
    if os.path.exists(runs):
        df = pd.read_csv(runs, dtype={"eps": str})
        fig_privacy_utility(df, out)
        fig_baselines(df, out)
        fig_gains(df, out)
        fig_ablation(df, out, a.ablation_eps)
    else:
        print("  run experiments/aggregate.py first for the summary figures")
    fig_pr_curves(a.results, out, a.fig_eps, a.seed)
    fig_mia_roc(a.results, out, a.fig_eps, a.seed)
    fig_sampler_scores(a.results, out, a.fig_eps, a.seed)
    for ds in DATASETS:
        fig_marginals(a.results, out, ds, a.fig_eps, a.seed, a.data_dir)
        fig_corrdiff(a.results, out, ds, a.fig_eps, a.seed, a.data_dir)
    fig_embeddings(a.results, out, a.fig_eps, a.seed, a.data_dir)


if __name__ == "__main__":
    main()
