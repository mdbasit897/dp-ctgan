#!/usr/bin/env python3
"""Aggregate results/**/metrics.json into CSV files, LaTeX tables and significance tests.

Outputs (results/summary/):
  runs.csv                 one row per (method/config, dataset, epsilon, seed)
  mean_std.csv             mean and std over seeds
  tab_utility.tex          TRTR / common / combined utility (manuscript Table "utility")
  tab_fidelity_privacy.tex fidelity + membership inference (manuscript Table "fidelity")
  tab_ablation.tex         ablation table
  tab_baselines.tex        ours vs baselines (readable names; non-private methods marked with a dagger)
  tab_baseline_minority.tex  synthetic minority rate of every baseline (degenerate class balance shows up here)
  tab_gate.tex             PrivacyGate near-copy / outlier rates (dp mode: audit only, nothing removed)
  tab_dp.tex               DP-SGD accounting of the discriminator (sigma, q, T, delta, epsilon)
  tab_confusion.tex        mean confusion matrices (RF, validation-selected threshold)
  stats.txt                Wilcoxon signed-rank tests, combined vs common (per-dataset p-values also
                           Holm-adjusted over the four datasets), and every ablation vs full
  tab_budget.tex           dp mode: epsilon per component, total (basic and RDP), moment noise, judge noise
  tab_nonprivate.tex       with --nonprivate-results: non-private variant vs RareSel-DP (RF AUPRC)

  python experiments/aggregate.py --results results
  python experiments/aggregate.py --results results_dp --baselines-dir results/baselines --nonprivate-results results
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import DATASET_LABELS, DATASETS

EPS_ORDER = ["nodp", "10", "3", "1"]
BASELINE_NAMES = {"sn_dpctgan": "DP-CTGAN (SmartNoise)", "sn_patectgan": "PATE-CTGAN (SmartNoise)",
                  "sn_mst": "MST (SmartNoise)", "ctgan": r"CTGAN$^\dagger$", "tvae": r"TVAE$^\dagger$",
                  "smote": r"SMOTE$^\dagger$", "pategan": "PATE-GAN", "dpgan": "DP-GAN", "aim": "AIM"}
BASELINE_NAMES.update({f"{m}_matched": BASELINE_NAMES[m].replace("(SmartNoise)", "(SmartNoise, matched)")
                       for m in ("sn_dpctgan", "sn_patectgan", "sn_mst")})
BASELINE_ORDER = ["sn_dpctgan_matched", "sn_patectgan_matched", "sn_mst_matched", "sn_dpctgan", "sn_patectgan",
                  "sn_mst", "pategan", "dpgan", "aim", "ctgan", "tvae", "smote"]
EPS_TEX = {"nodp": "No-DP", "10": r"$\varepsilon=10$", "3": r"$\varepsilon=3$", "1": r"$\varepsilon=1$",
           "5": r"$\varepsilon=5$"}


def flat(block, prefix=""):
    """Flatten the nested metrics of one config into {name: value}."""
    out = {}
    u = block.get("utility", {})
    for clf in ("rf", "knn"):
        for k, v in u.get(clf, {}).items():
            out[f"{clf}_{k}"] = v
    for k, v in block.get("fidelity", {}).items():
        out[k] = v
    pr = block.get("privacy", {})
    for att in ("knn_mia", "domias"):
        for k, v in pr.get(att, {}).items():
            out[f"{att}_{k}"] = v
    if "lr_mia" in pr:
        out["lr_mia_auc"] = pr["lr_mia"].get("lr_mia_auc")
        out["lr_mia_tpr@0.01"] = pr["lr_mia"].get("tpr_at_fpr_01")
    if "dcr" in pr:
        out["dcr_share"] = pr["dcr"]["share_closer_to_train"]
    return out


def eps_key(e):
    return "nodp" if e is None else f"{float(e):g}"


def collect(res, baselines_dir=None):
    baselines_dir = baselines_dir or os.path.join(res, "baselines")
    rows = []
    for f in glob.glob(os.path.join(res, "*", "eps*", "seed*", "metrics.json")):
        if os.sep + "baselines" + os.sep in f:
            continue
        R = json.load(open(f))
        m = R["meta"]
        base = {"method": "ours", "dataset": m["dataset"], "eps": eps_key(m["epsilon"]), "seed": m["seed"],
                "achieved_eps": m.get("achieved_epsilon"), "mode": m.get("mode", ""),
                "hp_source": m.get("hp_source"),
                **{f"dp_{k}": v for k, v in (m.get("dp") or {}).items()},
                **{f"budget_{k}": v for k, v in (m.get("dp_budget") or {}).items()},
                "total_min": R.get("timings_sec", {}).get("total", np.nan) / 60}
        rows.append({**base, "config": "trtr", **flat({"utility": R["trtr"]})})
        rare_counts = lambda blk: {f"rare_{k}": blk.get(k) for k in ("n_rare", "n_survivors", "n_selected",
                                                                       "n_padded") if blk.get(k) is not None}
        rows.append({**base, "config": "common", **flat(R["common"])})
        # n_selected / n_padded: rare records chosen by the judge vs plain samples added to reach n_r
        rows.append({**base, "config": "combined", **flat(R["combined"]),
                     **rare_counts({"n_rare": m.get("n_rare"), **(m.get("rare_sampler") or {})})})
        for name, blk in R.get("ablations", {}).items():
            rows.append({**base, "config": f"abl:{name}", **flat(blk), **rare_counts(blk)})
        for s, st in (R.get("privacy_gate") or {}).items():
            if st:
                rows.append({**base, "config": f"gate:{s}", **{f"gate_{k}": v for k, v in st.items()}})
    for f in glob.glob(os.path.join(baselines_dir, "*", "*", "eps*", "seed*", "metrics.json")):
        R = json.load(open(f))
        m = R["meta"]
        sn, delta_used = m.get("smartnoise") or {}, m.get("delta")
        if str(m.get("plugin", "")).startswith("sn_") and not sn:
            # SmartNoise runs with library defaults (no "smartnoise" metadata) store RareSel-DP's delta =
            # 1/(10 n_train), which is not passed to the library; recover n_train and report the library's own delta
            n_tr = 1.0 / (10.0 * float(m["delta"]))
            delta_used = 1e-9 if m["plugin"] == "sn_mst" else 1.0 / (n_tr * np.sqrt(n_tr))
            sn = {"preprocessor_eps": 0.1 * float(m["epsilon"])}
        rows.append({"method": m["method"], "config": m["method"], "dataset": m["dataset"],
                     "eps": eps_key(m["epsilon"]), "seed": m["seed"], "delta_used": delta_used,
                     "sn_preprocessor_eps": sn.get("preprocessor_eps"),
                     "minority_rate_synth": m.get("minority_rate_synth"), **flat(R)})
    return pd.DataFrame(rows)


def ms(series, digits=3):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "--"
    if len(s) == 1:
        return f"{s.iloc[0]:.{digits}f}"
    return f"{s.mean():.{digits}f}$\\pm${s.std(ddof=1):.{digits}f}"


def table(df, row_keys, cols, header, caption, label, bold_pairs=None):
    lines = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             r"\small", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{ll" + "c" * len(cols) + "}", r"\toprule", header + r" \\", r"\midrule"]
    prev_ds = None
    for ds, eps, cfg, label_txt in row_keys:
        sub = df[(df.dataset == ds) & (df.eps == eps) & (df.config == cfg)]
        if sub.empty:
            continue
        if prev_ds is not None and ds != prev_ds:
            lines.append(r"\midrule")
        cells = []
        for c in cols:
            txt = ms(sub[c]) if c in sub else "--"
            if bold_pairs and cfg in bold_pairs and c in sub:
                other = df[(df.dataset == ds) & (df.eps == eps) & (df.config == bold_pairs[cfg])]
                if c in other and not other.empty:
                    a, b = sub[c].mean(), other[c].mean()
                    lower_better = c in LOWER_BETTER
                    if (a < b) if lower_better else (a > b):
                        txt = r"\textbf{" + txt + "}"
            cells.append(txt)
        name = DATASET_LABELS.get(ds, ds) if ds != prev_ds else ""
        lines.append(f"{name} & {label_txt} & " + " & ".join(cells) + r" \\")
        prev_ds = ds
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


LOWER_BETTER = {"ks", "js", "ks_minority", "js_minority", "corr_diff", "disc_auc", "knn_mia_auc", "domias_auc",
                "lr_mia_auc", "domias_tpr@0.01", "knn_mia_tpr@0.01", "dcr_share"}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def eps_list(df, ds):
    present = df[df.dataset == ds].eps.unique().tolist()
    return [e for e in EPS_ORDER if e in present] + sorted(set(present) - set(EPS_ORDER))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--ablation-eps", default="3")
    ap.add_argument("--baselines-dir", default=None, help="default: <results>/baselines")
    ap.add_argument("--nonprivate-results", default=None,
                    help="results folder of the non-private variant (run_experiments.py --mode nonprivate); adds tab_nonprivate.tex")
    a = ap.parse_args()
    out = os.path.join(a.results, "summary")
    os.makedirs(out, exist_ok=True)
    df = collect(a.results, a.baselines_dir)
    if df.empty:
        raise SystemExit("no metrics.json found")
    df.to_csv(os.path.join(out, "runs.csv"), index=False)
    num = df.select_dtypes("number").columns.drop("seed", errors="ignore")
    g = df.groupby(["method", "config", "dataset", "eps"])[list(num)]
    pd.concat({"mean": g.mean(), "std": g.std(), "n": g.count()}, axis=1).to_csv(os.path.join(out, "mean_std.csv"))
    datasets = [d for d in DATASETS if d in df.dataset.unique()]
    pair = {"combined": "common", "common": "combined"}

    # utility
    keys = []
    for ds in datasets:
        e0 = eps_list(df, ds)[0]
        keys.append((ds, e0, "trtr", "TRTR"))
        for e in eps_list(df, ds):
            keys += [(ds, e, "common", f"{EPS_TEX.get(e, e)}, common"), (ds, e, "combined", f"{EPS_TEX.get(e, e)}, combined")]
    cols = ["rf_auc", "rf_auprc", "rf_f1", "knn_auc", "knn_auprc", "knn_f1"]
    hdr = r"Dataset & Configuration & RF AUC & RF AUPRC & RF F1 & $k$-NN AUC & $k$-NN AUPRC & $k$-NN F1"
    open(os.path.join(out, "tab_utility.tex"), "w").write(table(
        df[df.method == "ours"], keys, cols, hdr,
        "TSTR utility on the sealed test sets (mean $\\pm$ std over seeds). F1 at the threshold that maximizes F1 "
        "on the validation split. Bold: better of common and combined.", "tab:utility", pair))

    # fidelity + privacy
    keys = [k for k in keys if k[2] != "trtr"]
    # (the discriminator AUC is about 0.98-1.0 for every generator, baselines included, so it does not
    #  discriminate between methods; it stays in runs.csv / mean_std.csv but not in the table)
    cols = ["ks", "ks_minority", "js", "corr_diff", "knn_mia_auc", "domias_auc", "domias_tpr@0.01", "lr_mia_auc",
            "dcr_share"]
    hdr = (r"Dataset & Configuration & KS & KS (minority) & JS & Corr.\ diff. & kNN-MIA AUC & DOMIAS AUC & "
           r"DOMIAS TPR@1\% & LR-MIA AUC & DCR share")
    open(os.path.join(out, "tab_fidelity_privacy.tex"), "w").write(table(
        df[df.method == "ours"], keys, cols, hdr,
        "Fidelity (lower is better; KS (minority) compares the minority class only) and membership inference "
        "(AUC 0.5 and DCR share 0.5 = no measurable leakage); mean $\\pm$ std over seeds.", "tab:fidelity", pair))

    # ablation
    abl = df[(df.method == "ours") & (df.eps == a.ablation_eps)]
    variants = [("common", "Common only"), ("abl:plain_oversample", r"+ plain $G(z,c_{\min})$ samples"),
                ("abl:no_flow_refine", "w/o flow scoring + refinement"), ("abl:no_refine", "w/o refinement (Stage 3)"),
                ("abl:no_judge", "w/o classifier judge"), ("abl:no_maxmin", "w/o diversity selection"),
                ("abl:no_qm", "w/o quantile matching"),
                ("abl:judge_disc", "judge = DP discriminator (no extra budget)"),
                ("abl:judge_rf_nonprivate", "judge = non-private RF (reference)"),
                ("combined", "Full sampler")]
    variants = [v for v in variants if (abl.config == v[0]).any()]
    lines = [r"\begin{table}[t]", r"\centering",
             f"\\caption{{Ablation of the rare-minority sampler at {EPS_TEX.get(a.ablation_eps, a.ablation_eps)} "
             r"(RF AUPRC, mean $\pm$ std over seeds; every variant adds the same number of minority records).}",
             r"\label{tab:ablation}", r"\small", r"\begin{tabular}{l" + "c" * len(datasets) + "}", r"\toprule",
             "Variant & " + " & ".join(DATASET_LABELS[d] for d in datasets) + r" \\", r"\midrule"]
    for cfg, txt in variants:
        lines.append(txt + " & " + " & ".join(ms(abl[(abl.dataset == d) & (abl.config == cfg)]["rf_auprc"])
                                               for d in datasets) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out, "tab_ablation.tex"), "w").write("\n".join(lines))

    # baselines
    present = [m for m in df.method.unique() if m != "ours"]
    meths = [m for m in BASELINE_ORDER if m in present] + sorted(set(present) - set(BASELINE_ORDER))
    if meths:
        n_seeds = {m: int(df[df.method == m].groupby(["dataset", "eps"]).seed.nunique().max()) for m in ["ours"] + meths}
        keys = []
        for ds in datasets:
            for e in eps_list(df, ds):
                keys.append((ds, e, "combined", f"{EPS_TEX.get(e, e)}, ours"))
                keys += [(ds, e, m, f"{EPS_TEX.get(e, e)}, {BASELINE_NAMES.get(m, m)}") for m in meths]
        seeds_txt = ", ".join(f"{BASELINE_NAMES.get(m, m) if m != 'ours' else 'ours'}: {n_seeds[m]}"
                              for m in ["ours"] + meths)
        open(os.path.join(out, "tab_baselines.tex"), "w").write(table(
            df, keys, ["rf_auc", "rf_auprc", "rf_f1", "ks", "domias_auc", "dcr_share"],
            r"Dataset & Setting & RF AUC & RF AUPRC & RF F1 & KS & DOMIAS AUC & DCR share",
            "Comparison with baselines on identical splits and evaluation (mean $\\pm$ std over seeds; seeds per "
            f"setting: {seeds_txt}). (SmartNoise): library defaults, i.e.\\ the library's delta and 10\\% of "
            "$\\varepsilon$ spent on private column bounds; (SmartNoise, matched): the same delta and public column "
            "bounds as ours, the full $\\varepsilon$ for the generator. "
            r"$^\dagger$Not differentially private (listed under No-DP); SMOTE interpolates real training records "
            "and is a utility reference, not a private release.", "tab:baselines"))
        # synthetic minority rate (class balance of each generator's output)
        mr = df[df.method.isin(meths)]
        lines = [r"\begin{tabular}{ll" + "c" * len(meths) + "}", r"\toprule",
                 "Dataset & Setting & " + " & ".join(BASELINE_NAMES.get(m, m) for m in meths) + r" \\", r"\midrule"]
        for ds in datasets:
            for e in eps_list(mr, ds):
                cells = []
                for m in meths:
                    v = pd.to_numeric(mr[(mr.dataset == ds) & (mr.eps == e) & (mr.method == m)]["minority_rate_synth"])
                    cells.append(f"{v.mean():.4f}" if v.notna().any() else "--")
                lines.append(f"{DATASET_LABELS[ds]} & {EPS_TEX.get(e, e)} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(out, "tab_baseline_minority.tex"), "w").write("\n".join(lines))

    # PrivacyGate and DP accounting
    dp_mode = (df[df.method == "ours"]["mode"] == "dp").any() if "mode" in df else False
    cap = (r"\caption{PrivacyGate audit: fraction of synthetic records that the gate flags as near-copies of training "
           r"records or as outliers (mean over seeds). In RareSel-DP the gate only audits; no record is "
           r"removed.}" if dp_mode else
           r"\caption{PrivacyGate statistics (fraction of candidate records kept, removed as near-copies, removed as "
           r"outliers; mean over seeds).}")
    keep_lbl = "Unflagged" if dp_mode else "Keep"
    lines = [r"\begin{table}[t]", r"\centering", cap, r"\label{tab:gate}", r"\small",
             r"\begin{tabular}{llcccccc}", r"\toprule",
             r" & & \multicolumn{3}{c}{Common stream} & \multicolumn{3}{c}{Rare stream} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             f"Dataset & Setting & {keep_lbl} & Near-copy & Outlier & {keep_lbl} & Near-copy & Outlier \\\\",
             r"\midrule"]
    gate = df[df.config.str.startswith("gate:")]
    for ds in datasets:
        for e in eps_list(df, ds):
            c = gate[(gate.dataset == ds) & (gate.eps == e) & (gate.config == "gate:common")]
            r = gate[(gate.dataset == ds) & (gate.eps == e) & (gate.config == "gate:rare")]
            if c.empty:
                continue
            f = lambda s, k: f"{pd.to_numeric(s[k]).mean():.3f}" if k in s and not s.empty else "--"
            lines.append(f"{DATASET_LABELS[ds]} & {EPS_TEX.get(e, e)} & " + " & ".join(
                [f(c, "gate_keep_rate"), f(c, "gate_near_copy_rate"), f(c, "gate_outlier_rate"),
                 f(r, "gate_keep_rate"), f(r, "gate_near_copy_rate"), f(r, "gate_outlier_rate")]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out, "tab_gate.tex"), "w").write("\n".join(lines))

    # DP-SGD accounting of the discriminator. dp mode: delta and epsilon of the discriminator's share of the
    # budget (the totals over all components are in tab_budget.tex); nonprivate mode: the discriminator is the whole budget.
    dp = df[(df.method == "ours") & (df.config == "combined") & (df.eps != "nodp")]
    if dp_mode:
        hdr_dp = (r"Dataset & Target $\varepsilon$ & $\sigma$ & $q$ & $T$ & $\delta_{\mathrm{GAN}}$ & "
                  r"$\varepsilon_{\mathrm{GAN}}$ \\")
        cols_dp = [("dp_noise_multiplier", "{:.3f}"), ("dp_sample_rate", "{:.4f}"), ("dp_steps_done", "{:.0f}"),
                   ("budget_delta_each", "{:.1e}"), ("budget_epsilon_gan_achieved", "{:.3f}")]
    else:
        hdr_dp = r"Dataset & Target $\varepsilon$ & $\sigma$ & $q$ & $T$ & $\delta$ & Achieved $\varepsilon$ \\"
        cols_dp = [("dp_noise_multiplier", "{:.3f}"), ("dp_sample_rate", "{:.4f}"), ("dp_steps_done", "{:.0f}"),
                   ("dp_delta", "{:.1e}"), ("achieved_eps", "{:.3f}")]
    lines = [r"\begin{tabular}{llccccc}", r"\toprule", hdr_dp, r"\midrule"]
    for ds in datasets:
        for e in [x for x in EPS_ORDER if x != "nodp"]:
            s = dp[(dp.dataset == ds) & (dp.eps == e)]
            if s.empty:
                continue
            cells = [(fmt.format(pd.to_numeric(s[k]).mean()) if k in s and s[k].notna().any() else "--")
                     for k, fmt in cols_dp]
            lines.append(f"{DATASET_LABELS[ds]} & {e} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "tab_dp.tex"), "w").write("\n".join(lines))

    # confusion matrices (mean counts over seeds)
    lines = [r"\begin{tabular}{llrrrrccc}", r"\toprule",
             r"Dataset & Configuration & TN & FP & FN & TP & Precision & Recall & F1 \\", r"\midrule"]
    for ds in datasets:
        for e in eps_list(df, ds):
            for cfg in (["trtr"] if e == eps_list(df, ds)[0] else []) + ["common", "combined"]:
                s = df[(df.method == "ours") & (df.dataset == ds) & (df.eps == e) & (df.config == cfg)]
                if s.empty:
                    continue
                lab = "TRTR" if cfg == "trtr" else f"{EPS_TEX.get(e, e)}, {cfg}"
                lines.append(f"{DATASET_LABELS[ds]} & {lab} & " + " & ".join(
                    f"{s[f'rf_{k}'].mean():,.0f}" for k in ("tn", "fp", "fn", "tp")) + " & " + " & ".join(
                    ms(s[f"rf_{k}"]) for k in ("precision", "recall", "f1")) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "tab_confusion.tex"), "w").write("\n".join(lines))

    # dp mode: privacy budget per component
    bud = df[(df.method == "ours") & (df.config == "combined") & (df.eps != "nodp")]
    if "budget_epsilon_total_rdp" in bud:
        cols_b = [("budget_epsilon_gan_achieved", "{:.2f}"), ("budget_epsilon_stats", "{:.2f}"),
                  ("budget_epsilon_judge_achieved", "{:.2f}"), ("budget_epsilon_total_basic", "{:.2f}"),
                  ("budget_epsilon_total_rdp", "{:.2f}"), ("budget_stats_rel_noise", "{:.3f}"),
                  ("budget_mm_weight", "{:.2f}"), ("budget_judge_noise_multiplier", "{:.2f}")]
        lines = [r"\begin{tabular}{ll" + "c" * len(cols_b) + "}", r"\toprule",
                 r"Dataset & $\varepsilon$ & $\varepsilon_{\mathrm{GAN}}$ & $\varepsilon_{\mathrm{stats}}$ & "
                 r"$\varepsilon_{\mathrm{judge}}$ & Total (basic) & Total (RDP) & Moment noise/$n$ & "
                 r"MM weight & Judge $\sigma$ \\", r"\midrule"]
        for ds in datasets:
            for e in [x for x in EPS_ORDER if x != "nodp"]:
                s_ = bud[(bud.dataset == ds) & (bud.eps == e)]
                if s_.empty:
                    continue
                cells = [(fmt.format(pd.to_numeric(s_[c]).mean()) if c in s_ and s_[c].notna().any() else "--")
                         for c, fmt in cols_b]
                lines.append(f"{DATASET_LABELS[ds]} & {e} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(out, "tab_budget.tex"), "w").write("\n".join(lines))

    # non-private variant vs RareSel-DP
    if a.nonprivate_results:
        leg = collect(a.nonprivate_results, os.path.join(a.nonprivate_results, "_none_"))
        leg = leg[leg.method == "ours"]
        ours_ = df[df.method == "ours"]
        lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
                 r" & & \multicolumn{3}{c}{Non-private variant (upper reference)} & "
                 r"\multicolumn{3}{c}{RareSel-DP} \\",
                 r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
                 r"Dataset & $\varepsilon$ & Common & Combined & Gain & Common & Combined & Gain \\", r"\midrule"]
        for ds in datasets:
            for e in eps_list(df, ds):
                g = lambda d_, c: pd.to_numeric(d_[(d_.dataset == ds) & (d_.eps == e) & (d_.config == c)]["rf_auprc"])
                lc, lb, pc, pb = g(leg, "common"), g(leg, "combined"), g(ours_, "common"), g(ours_, "combined")
                if pc.empty:
                    continue
                gain = lambda b_, c_: f"{b_.mean() - c_.mean():+.3f}" if len(b_) and len(c_) else "--"
                lines.append(f"{DATASET_LABELS[ds]} & {EPS_TEX.get(e, e)} & {ms(lc)} & {ms(lb)} & {gain(lb, lc)} & "
                             f"{ms(pc)} & {ms(pb)} & {gain(pb, pc)} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(out, "tab_nonprivate.tex"), "w").write("\n".join(lines))

    # significance: combined vs common, paired over (dataset, eps, seed)
    ours = df[df.method == "ours"]
    a_ = ours[ours.config == "combined"].set_index(["dataset", "eps", "seed"])
    b_ = ours[ours.config == "common"].set_index(["dataset", "eps", "seed"])
    common_idx = a_.index.intersection(b_.index)
    rep = ["Wilcoxon signed-rank test, combined vs common, paired over (dataset, eps, seed)", ""]
    for m in ["rf_auc", "rf_auprc", "rf_f1", "knn_auc", "knn_auprc", "knn_f1", "ks", "js", "domias_auc", "lr_mia_auc"]:
        if m not in a_:
            continue
        x = pd.to_numeric(a_.loc[common_idx, m]); y = pd.to_numeric(b_.loc[common_idx, m])
        d = (x - y).dropna()
        if len(d) < 6 or (d == 0).all():
            continue
        p = wilcoxon(d).pvalue
        rep.append(f"{m:12s} n={len(d):3d}  mean diff={d.mean():+.4f}  combined better in "
                   f"{(d < 0).sum() if m in LOWER_BETTER else (d > 0).sum()}/{len(d)}  p={p:.2e}")
        per_ds = []
        for ds in datasets:
            dd = d.xs(ds, level="dataset") if ds in d.index.get_level_values(0) else pd.Series(dtype=float)
            if len(dd) >= 6 and not (dd == 0).all():
                per_ds.append((ds, dd, wilcoxon(dd).pvalue))
        for (ds, dd, p_), p_adj in zip(per_ds, holm([x[2] for x in per_ds])):
            rep.append(f"    {ds:14s} n={len(dd):2d} mean diff={dd.mean():+.4f} p={p_:.2e}  p_holm={p_adj:.2e}")

    # primary test on the DP runs only (No-DP excluded); robustness: without seed 0 (the tuning seed) and
    # with seeds as units (runs with the same seed share a split, so they are not independent)
    def wil(d, label):
        d = d.dropna()
        if len(d) >= 6 and not (d == 0).all():
            rep.append(f"  {label:38s} n={len(d):3d} mean diff={d.mean():+.4f} combined better in "
                       f"{int((d > 0).sum())}/{len(d)}  p={wilcoxon(d).pvalue:.2e}")
    rep += ["", "DP runs only (eps in {1, 3, 10}), combined - common"]
    for m in ["rf_auprc", "knn_auprc", "rf_auc"]:
        if m not in a_:
            continue
        d = pd.to_numeric(a_.loc[common_idx, m]) - pd.to_numeric(b_.loc[common_idx, m])
        dp = d[d.index.get_level_values("eps") != "nodp"]
        seed_lvl = dp.groupby(level=["dataset", "seed"]).mean()
        rep.append(f"[{m}]")
        wil(dp, "all DP runs")
        wil(dp[dp.index.get_level_values("seed") != 0], "DP runs, seeds 1-4 (tuning seed left out)")
        wil(seed_lvl, "dataset-seed units (mean over budgets)")
        per_ds = [(ds, dp.xs(ds, level="dataset")) for ds in datasets if ds in dp.index.get_level_values(0)]
        per_ds = [(ds, dd, wilcoxon(dd).pvalue) for ds, dd in per_ds if len(dd) >= 6 and not (dd == 0).all()]
        for (ds, dd, p_), p_adj in zip(per_ds, holm([x[2] for x in per_ds])):
            sl = seed_lvl.xs(ds, level="dataset")
            rep.append(f"    {ds:14s} n={len(dd):2d} mean diff={dd.mean():+.4f} p={p_:.2e}  p_holm={p_adj:.2e}  "
                       f"seeds with a positive mean gain: {int((sl > 0).sum())}/{len(sl)}")

    # rare streams that the judge could not fill (completed with plain G(z, c_min) samples)
    if "rare_n_padded" in ours:
        rep += ["", "Rare streams completed with plain samples (judge accepted fewer than n_r candidates)"]
        for cfg in ["combined"] + sorted(c for c in ours.config.unique() if c.startswith("abl:")):
            x = ours[ours.config == cfg]
            pad = pd.to_numeric(x.get("rare_n_padded"), errors="coerce")
            if pad.notna().sum() == 0:
                continue
            rep.append(f"  {cfg:28s} padded in {int((pad > 0).sum())}/{int(pad.notna().sum())} runs")
            if cfg == "combined":
                for _, r in x[pad > 0].sort_values(["dataset", "eps", "seed"]).iterrows():
                    rep.append(f"      {r.dataset} eps={r.eps} seed={r.seed}: judge-selected {int(r.rare_n_selected)}, "
                               f"padded {int(r.rare_n_padded)} of {int(r.rare_n_rare)}")
    full_ = ours[ours.config == "combined"].set_index(["dataset", "eps", "seed"])
    abl_cfgs = sorted(c for c in ours.config.unique() if c.startswith("abl:"))
    if abl_cfgs:
        rep += ["", "Ablations: variant - full sampler, paired over (dataset, eps, seed); negative = stage helps", ""]
        for m in ["rf_auprc", "rf_auc", "knn_auprc", "rf_f1"]:
            rep.append(f"[{m}]")
            for v in abl_cfgs:
                x = ours[ours.config == v].set_index(["dataset", "eps", "seed"])
                if m not in x:
                    continue
                d = (pd.to_numeric(x[m]) - pd.to_numeric(full_[m])).dropna()
                if len(d) < 6 or (d == 0).all():
                    continue
                per = " ".join(f"{ds[:6]}:{d.xs(ds, level=0).mean():+.3f}" for ds in datasets
                               if ds in d.index.get_level_values(0))
                rep.append(f"  {v[4:]:22s} n={len(d):3d} mean={d.mean():+.4f} worse-than-full={int((d < 0).sum()):3d} "
                           f"p={wilcoxon(d).pvalue:.1e}   {per}")
    open(os.path.join(out, "stats.txt"), "w").write("\n".join(rep) + "\n")
    print("\n".join(rep))
    print(f"\nwrote tables and CSVs to {out}/")


if __name__ == "__main__":
    main()
