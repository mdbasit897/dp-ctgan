#!/usr/bin/env python3
"""Publication figures from the experiment results (vector PDF + PNG preview).

  python experiments/make_figures.py --results results --fig-eps 3

Figures (results/figures/):
  fig_privacy_utility.pdf   RF AUPRC vs privacy budget, common vs combined, real-data reference
  fig_gains.pdf             combined - common difference with 95% CI, per dataset and budget
  fig_ablation.pdf          ablation of the sampler (needs --ablations runs)
  fig_pr_curves.pdf         test precision-recall curves: real data, common, combined (seed 0)
  fig_mia_roc.pdf           DOMIAS-style attack ROC on log-FPR axes, common vs combined (seed 0)
  fig_sampler_scores.pdf    flow likelihood-ratio scores: pool, before/after refinement, selected
  fig_marginals_<ds>.pdf    minority-class marginals: real vs common vs rare stream
  fig_embedding_<ds>.pdf    t-SNE of real minority, common-stream minority and rare stream
  fig_corrdiff_<ds>.pdf     |corr(real) - corr(synthetic)| for common and combined

Colours follow a colour-blind-validated categorical palette; every series also has a
distinct marker / line style so figures stay readable in greyscale print.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler

from common import DATASET_LABELS, DATASETS, load, make_splits

# ---- style -------------------------------------------------------------------------------------
COMBINED, COMMON, REAL = "#2a78d6", "#eb6834", "#52514e"   # blue, orange, neutral grey (reference)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                     "#256abf", "#184f95", "#0d366b"])
STYLE = {"common": dict(color=COMMON, marker="o", ls="--", label="Common only"),
         "combined": dict(color=COMBINED, marker="s", ls="-", label="Combined (with rare stream)"),
         "trtr": dict(color=REAL, ls=":", label="Real data (TRTR)")}
EPS_POS = ["1", "3", "10", "nodp"]
EPS_LBL = {"1": "1", "3": "3", "10": "10", "nodp": "No DP"}
FULL_W, HALF_W = 6.8, 3.3   # inches (ACM single column text width ~ 6.8 in)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.5, "lines.markersize": 4.5, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300,
})


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=200)
    plt.close(fig)
    print("  wrote", name)


def ci95(x):
    x = np.asarray(pd.to_numeric(x, errors="coerce").dropna())
    if len(x) < 2:
        return (x.mean() if len(x) else np.nan), 0.0
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


# ---- figures from runs.csv -----------------------------------------------------------------
def fig_privacy_utility(df, out, metric="rf_auprc", ylabel="RF AUPRC (test)"):
    ds_list = [d for d in DATASETS if d in df.dataset.unique()]
    fig, axes = plt.subplots(1, len(ds_list), figsize=(FULL_W, 2.0), squeeze=False)
    for ax, ds in zip(axes[0], ds_list):
        sub = df[(df.method == "ours") & (df.dataset == ds)]
        pos = [e for e in EPS_POS if e in sub.eps.unique()]
        xs = np.arange(len(pos))
        for cfg in ("common", "combined"):
            m = [sub[(sub.eps == e) & (sub.config == cfg)][metric].mean() for e in pos]
            s = [sub[(sub.eps == e) & (sub.config == cfg)][metric].std() for e in pos]
            st = STYLE[cfg]
            ax.errorbar(xs, m, yerr=np.nan_to_num(s), color=st["color"], marker=st["marker"], ls=st["ls"],
                        capsize=2, elinewidth=0.8, label=st["label"])
        trtr = sub[sub.config == "trtr"][metric].mean()
        ax.axhline(trtr, **{k: v for k, v in STYLE["trtr"].items()}, lw=1.2)
        ax.set_xticks(xs, [EPS_LBL[e] for e in pos])
        ax.set_title(DATASET_LABELS[ds], color=INK)
        ax.set_xlabel(r"Privacy budget $\varepsilon$")
    axes[0][0].set_ylabel(ylabel)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    save(fig, out, "fig_privacy_utility")


def fig_gains(df, out, metrics=(("rf_auprc", "RF AUPRC"), ("knn_auprc", "$k$-NN AUPRC"), ("rf_auc", "RF AUC"))):
    ours = df[df.method == "ours"]
    a = ours[ours.config == "combined"].set_index(["dataset", "eps", "seed"])
    b = ours[ours.config == "common"].set_index(["dataset", "eps", "seed"])
    idx = a.index.intersection(b.index)
    rows = [(d, e) for d in DATASETS for e in EPS_POS if (d, e) in {(i[0], i[1]) for i in idx}]
    fig, axes = plt.subplots(1, len(metrics), figsize=(FULL_W, 0.28 * len(rows) + 0.8), sharey=True, squeeze=False)
    for ax, (m, title) in zip(axes[0], metrics):
        d = (pd.to_numeric(a.loc[idx, m]) - pd.to_numeric(b.loc[idx, m]))
        for i, (ds, e) in enumerate(rows):
            mean, h = ci95(d.xs((ds, e), level=("dataset", "eps")))
            ax.errorbar(mean, i, xerr=h, fmt="o", color=COMBINED, ms=4, capsize=2, elinewidth=0.9)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(title, color=INK)
        ax.set_xlabel("Combined $-$ common")
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(range(len(rows)), [f"{DATASET_LABELS[d]}, {EPS_LBL[e]}" for d, e in rows])
    axes[0][0].invert_yaxis()
    fig.tight_layout()
    save(fig, out, "fig_gains")


def fig_ablation(df, out, eps="3", metric="rf_auprc"):
    sub = df[(df.method == "ours") & (df.eps == eps)]
    variants = [("common", "Common only"), ("abl:plain_oversample", "+ plain minority samples"),
                ("abl:no_flow_refine", "w/o flow + refinement"), ("abl:no_refine", "w/o refinement"),
                ("abl:no_judge", "w/o judge"), ("abl:no_maxmin", "w/o diversity"),
                ("abl:no_qm", "w/o quantile matching"), ("combined", "Full sampler")]
    variants = [v for v in variants if (sub.config == v[0]).any()]
    if len(variants) < 3:
        print("  (no ablation runs found; skipping fig_ablation)")
        return
    ds_list = [d for d in DATASETS if d in sub.dataset.unique()]
    fig, axes = plt.subplots(1, len(ds_list), figsize=(FULL_W, 0.25 * len(variants) + 0.9), sharey=True,
                             squeeze=False)
    for ax, ds in zip(axes[0], ds_list):
        for i, (cfg, _) in enumerate(variants):
            mean, h = ci95(sub[(sub.dataset == ds) & (sub.config == cfg)][metric])
            c = COMBINED if cfg == "combined" else (COMMON if cfg == "common" else REAL)
            ax.errorbar(mean, i, xerr=h, fmt="o", color=c, ms=4, capsize=2, elinewidth=0.9)
        ax.set_title(DATASET_LABELS[ds], color=INK)
        ax.set_xlabel("RF AUPRC")
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(range(len(variants)), [v[1] for v in variants])
    axes[0][0].invert_yaxis()
    fig.tight_layout()
    save(fig, out, "fig_ablation")


# ---- figures from seed artefacts ---------------------------------------------------------------
def art(results, ds, eps, seed, name):
    p = os.path.join(results, ds, f"eps{eps}", f"seed{seed}", name)
    return p if os.path.exists(p) else None


def fig_pr_curves(results, out, eps, seed):
    ds_list = [d for d in DATASETS if art(results, d, eps, seed, "probs.npz")]
    if not ds_list:
        return
    fig, axes = plt.subplots(1, len(ds_list), figsize=(FULL_W, 2.0), squeeze=False)
    for ax, ds in zip(axes[0], ds_list):
        z = np.load(art(results, ds, eps, seed, "probs.npz"))
        y = z["y_test"]
        for key, cfg in (("trtr_rf", "trtr"), ("common_rf", "common"), ("combined_rf", "combined")):
            p, r, _ = precision_recall_curve(y, z[key])
            st = STYLE[cfg]
            ax.plot(r, p, color=st["color"], ls=st["ls"], lw=1.3, label=st["label"])
        ax.axhline(y.mean(), color=GRID, lw=1)
        ax.set_title(DATASET_LABELS[ds], color=INK)
        ax.set_xlabel("Recall"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("Precision")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    save(fig, out, "fig_pr_curves")


def fig_mia_roc(results, out, eps, seed, attack="domias"):
    ds_list = [d for d in DATASETS if art(results, d, eps, seed, "mia_scores.npz")]
    if not ds_list:
        return
    fig, axes = plt.subplots(1, len(ds_list), figsize=(FULL_W, 2.0), squeeze=False)
    for ax, ds in zip(axes[0], ds_list):
        z = np.load(art(results, ds, eps, seed, "mia_scores.npz"))
        for cfg in ("common", "combined"):
            fpr, tpr, _ = roc_curve(z[f"{cfg}_y"], z[f"{cfg}_{attack}"])
            st = STYLE[cfg]
            ax.plot(np.clip(fpr, 1e-4, 1), tpr, color=st["color"], ls=st["ls"], lw=1.3, label=st["label"])
        g = np.logspace(-4, 0, 50)
        ax.plot(g, g, color=MUTED, lw=0.8, ls=":", label="Random guess")
        ax.set_xscale("log"); ax.set_xlim(1e-3, 1); ax.set_ylim(0, 1)
        ax.set_title(DATASET_LABELS[ds], color=INK); ax.set_xlabel("False-positive rate")
    axes[0][0].set_ylabel("True-positive rate")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    save(fig, out, f"fig_mia_roc_{attack}")


def fig_sampler_scores(results, out, eps, seed):
    ds_list = [d for d in DATASETS if art(results, d, eps, seed, "sampler_diag.npz")]
    if not ds_list:
        return
    fig, axes = plt.subplots(1, len(ds_list), figsize=(FULL_W, 1.9), squeeze=False)
    for ax, ds in zip(axes[0], ds_list):
        z = np.load(art(results, ds, eps, seed, "sampler_diag.npz"))
        series = [("pool_score", "Stage 1 pool", REAL, "-"), ("refined_before", "Before refinement", COMMON, "--"),
                  ("refined_after", "After refinement", COMBINED, "-"), ("selected_score", "Selected", INK, ":")]
        allv = np.concatenate([z[k] for k, *_ in series if k in z])
        lo, hi = np.percentile(allv, [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        for k, lab, c, ls in series:
            if k in z and len(z[k]):
                ax.hist(z[k][(z[k] >= lo) & (z[k] <= hi)], bins=bins, density=True, histtype="step", color=c, ls=ls, lw=1.3,
                        label=lab)
        ax.set_title(DATASET_LABELS[ds], color=INK)
        ax.set_xlabel(r"$\log p_{\min}(h) - \log p_{\mathrm{maj}}(h)$")
        ax.set_yticks([])
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
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
    fig, axes = plt.subplots(1, len(sep), figsize=(FULL_W, 1.8), squeeze=False)
    cmin = common[common[target] == 1]
    for ax, c in zip(axes[0], sep):
        lo, hi = np.percentile(np.r_[rmin[c], cmin[c], rare[c]], [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        inr = lambda v: v[(v >= lo) & (v <= hi)]
        ax.hist(inr(rmin[c]), bins=bins, density=True, color=REAL, alpha=0.25, label="Real minority")
        ax.hist(inr(cmin[c]), bins=bins, density=True, histtype="step", color=COMMON, ls="--", lw=1.3,
                label="Common-stream minority")
        ax.hist(inr(rare[c]), bins=bins, density=True, histtype="step", color=COMBINED, lw=1.3,
                label="Rare stream")
        ax.set_title(c, color=INK); ax.set_yticks([])
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    save(fig, out, f"fig_marginals_{ds}")


def fig_embedding(results, out, ds, eps, seed, data_dir, n=1000):
    got = _real_and_synth(results, ds, eps, seed, data_dir)
    if got is None:
        return
    train, common, rare, target, _ = got
    feat = [c for c in train.columns if c != target]
    rng = np.random.RandomState(0)
    take = lambda d: d.iloc[rng.choice(len(d), min(n, len(d)), replace=False)][feat].values.astype(float)
    groups = [("Real minority", take(train[train[target] == 1]), REAL, "o"),
              ("Common-stream minority", take(common[common[target] == 1]), COMMON, "^"),
              ("Rare stream", take(rare), COMBINED, "s")]
    X = StandardScaler().fit(groups[0][1]).transform(np.vstack([g[1] for g in groups]))
    Z = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(X)
    fig, ax = plt.subplots(figsize=(HALF_W, HALF_W))
    i = 0
    for lab, G, c, mk in groups:
        ax.scatter(Z[i:i + len(G), 0], Z[i:i + len(G), 1], s=6, color=c, marker=mk, alpha=0.55, lw=0,
                   label=lab)
        i += len(G)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(f"{DATASET_LABELS[ds]} (t-SNE)", color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, markerscale=2)
    save(fig, out, f"fig_embedding_{ds}")


def fig_corrdiff(results, out, ds, eps, seed, data_dir):
    got = _real_and_synth(results, ds, eps, seed, data_dir)
    if got is None:
        return
    train, common, rare, target, _ = got
    feat = [c for c in train.columns if c != target]
    combined = pd.concat([common, rare], ignore_index=True)
    cr = train[feat].corr().fillna(0).values
    mats = [("Common only", np.abs(cr - common[feat].corr().fillna(0).values)),
            ("Combined", np.abs(cr - combined[feat].corr().fillna(0).values))]
    vmax = max(m.max() for _, m in mats)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W * 0.75, FULL_W * 0.38))
    for ax, (t, m) in zip(axes, mats):
        im = ax.imshow(m, cmap=SEQ, vmin=0, vmax=vmax)
        ax.set_title(f"{t}: mean {m[np.triu_indices(len(feat), 1)].mean():.3f}", color=INK)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(r"$|\rho_{\mathrm{real}} - \rho_{\mathrm{synth}}|$")
    cb.outline.set_visible(False)
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
    runs = os.path.join(a.results, "summary", "runs.csv")
    if os.path.exists(runs):
        df = pd.read_csv(runs, dtype={"eps": str})
        fig_privacy_utility(df, out)
        fig_gains(df, out)
        fig_ablation(df, out, a.ablation_eps)
    else:
        print("  run experiments/aggregate.py first for the summary figures")
    fig_pr_curves(a.results, out, a.fig_eps, a.seed)
    fig_mia_roc(a.results, out, a.fig_eps, a.seed)
    fig_sampler_scores(a.results, out, a.fig_eps, a.seed)
    for ds in DATASETS:
        fig_marginals(a.results, out, ds, a.fig_eps, a.seed, a.data_dir)
        fig_embedding(a.results, out, ds, a.fig_eps, a.seed, a.data_dir)
        fig_corrdiff(a.results, out, ds, a.fig_eps, a.seed, a.data_dir)


if __name__ == "__main__":
    main()
