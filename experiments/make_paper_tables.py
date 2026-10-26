"""Build the LaTeX tables of the manuscript from the experiment outputs.

Reads (relative to --root, the repository root by default)
  results_dp/summary/*                    aggregate.py output of the main grid (tuned hyperparameters)
  results_dp/<ds>/eps<e>/seed<s>/metrics.json   accounting details (judge, minority statistics)
  results_dp_untuned/summary/runs.csv     the same grid with the dataset presets (--hp-dir none)
  results/hparams/<ds>_eps<e>.json        selected hyperparameters (tune.py)
and writes one .tex file per table to --out.

  python experiments/aggregate.py --results results_dp --baselines-dir results/baselines \\
      --nonprivate-results results --ablation-eps 3
  python experiments/aggregate.py --results results_dp_untuned --ablation-eps 3
  python experiments/make_paper_tables.py --out paper_tables
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DS = ["adult", "taiwan_credit", "credit_fraud", "mimic3"]
LBL = {"adult": "Adult", "taiwan_credit": "Taiwan", "credit_fraud": "Credit Card Fraud", "mimic3": "MIMIC-III"}
ETX = {"1": r"$\varepsilon=1$", "3": r"$\varepsilon=3$", "10": r"$\varepsilon=10$", "nodp": "No-DP"}
# dataset presets of dp_ctgan_pipeline.HP_PRESETS / Hyperparams (used by --hp-dir none and as tuning defaults)
PRESET = {"adult": dict(gan_epochs=80, dp_max_grad_norm=5.0, r1_gamma=5.0),
          "taiwan_credit": dict(gan_epochs=80, dp_max_grad_norm=5.0, r1_gamma=10.0),
          "credit_fraud": dict(gan_epochs=50, dp_max_grad_norm=1.0, r1_gamma=10.0),
          "mimic3": dict(gan_epochs=100, dp_max_grad_norm=1.0, r1_gamma=10.0)}
PRESET_ALL = dict(batch_size=512, gen_lr=2e-4, disc_lr=2e-4)
SN_NOTE = (r"DP-CTGAN, PATE-CTGAN and MST are the SmartNoise implementations run with the same $\delta$ as \method{}, "
           r"public column bounds taken from the training split (as for \method{}'s encoder), the whole $\varepsilon$ "
           r"for the generator, and 5 seeds; results with the library defaults are in Table~\ref{tab:baselines-full}.")


def ms(s, k=3):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return "--"
    return f"{s.mean():.{k}f}" if len(s) == 1 else f"{s.mean():.{k}f}$\\pm${s.std(ddof=1):.{k}f}"


def pfmt(p):
    return f"{p:.1e}".replace("e-0", "e-").replace("e+00", "")


def write(out, name, lines):
    with open(os.path.join(out, name), "w") as f:
        f.write("\n".join(lines) + "\n")


def set_caption(t, caption=None, label=None):
    if caption is not None:
        t = re.sub(r"\\caption\{.*?\}\n\\label", lambda _: "\\caption{" + caption + "}\n\\label", t, count=1, flags=re.S)
    if label is not None:
        t = re.sub(r"\\label\{[^}]*\}", lambda _: "\\label{" + label + "}", t, count=1)
    return t


def wrap(body, caption, label, place="htbp", wide=False):
    if wide:
        body = "\\resizebox{\\textwidth}{!}{%\n" + body.rstrip("\n") + "}"
    return ["\\begin{table}[" + place + "]", "\\centering", "\\caption{" + caption + "}", "\\label{" + label + "}",
            "\\small", body.rstrip("\n"), "\\end{table}"]


def run_meta(root, ds, e):
    """Accounting details per (dataset, eps), averaged over the saved seeds."""
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "results_dp", ds, f"eps{e}", "seed*", "metrics.json"))):
        m = json.load(open(f))["meta"]
        dp, j, s = m["dp"], m["judge"], m["minority_stats"]
        rows.append(dict(C=dp["max_grad_norm"], sig_d=dp["noise_multiplier"], q_d=dp["sample_rate"],
                         T_d=dp["steps_planned"], eps_d=dp["achieved_epsilon"], sig_j=j["noise_multiplier"],
                         q_j=j["sample_rate"], T_j=j["steps"], delta2=s["sensitivity"],
                         sig_s=s["noise_multiplier"] * s["sensitivity"], delta=m["dp_budget"]["delta"],
                         n_train=m["n_train"]))
    return pd.DataFrame(rows).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="directory that contains results_dp/, results_dp_untuned/, results/")
    ap.add_argument("--results", default="results_dp")
    ap.add_argument("--untuned", default="results_dp_untuned")
    ap.add_argument("--hparams", default="results/hparams")
    ap.add_argument("--out", default="paper_tables")
    args = ap.parse_args()
    R = lambda *p: os.path.join(args.root, *p)
    SRC = R(args.results, "summary")
    os.makedirs(args.out, exist_ok=True)
    d = pd.read_csv(os.path.join(SRC, "runs.csv"), dtype={"eps": str})
    u = pd.read_csv(R(args.untuned, "summary", "runs.csv"), dtype={"eps": str})
    ours, uo = d[d.method == "ours"], u[u.method == "ours"]
    key = ["dataset", "eps", "seed"]

    # ---- Table: main-text baseline comparison (RF AUPRC) ---------------------------------
    cols = [("ours", "combined", r"\method{}", d), ("ours", "combined", r"\method{} (untuned)", u),
            ("sn_dpctgan_matched", None, "DP-CTGAN", d), ("sn_patectgan_matched", None, "PATE-CTGAN", d),
            ("sn_mst_matched", None, "MST", d), ("ctgan", None, r"CTGAN$^\dagger$", d),
            ("tvae", None, r"TVAE$^\dagger$", d), ("smote", None, r"SMOTE$^\dagger$", d)]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{RF AUPRC on the sealed test sets for \method{} (combined data; tuned hyperparameters and dataset "
         r"presets) and the baselines (mean $\pm$ std; 5 seeds, CTGAN and TVAE 3 seeds). " + SN_NOTE +
         r" $^\dagger$Not differentially private; SMOTE interpolates real training records and is a utility "
         r"reference, not a private release. Best DP result per row in bold.}",
         r"\label{tab:baselines}", r"\small", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{ll" + "c" * len(cols) + "}", r"\toprule",
         r" & & \multicolumn{2}{c}{\method{}} & \multicolumn{3}{c}{SmartNoise (matched)} & \multicolumn{3}{c}{Non-private references} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
         r"Dataset & Setting & Tuned & Untuned & DP-CTGAN & PATE-CTGAN & MST & CTGAN$^\dagger$ & TVAE$^\dagger$ & SMOTE$^\dagger$ \\",
         r"\midrule"]
    for i, ds in enumerate(DS):
        if i:
            L.append(r"\midrule")
        for e in ["1", "3", "10", "nodp"]:
            cells, means = [], []
            for m, cfg, _, src in cols:
                s = src[(src.method == m) & (src.dataset == ds) & (src.eps == e)]
                if cfg:
                    s = s[s.config == cfg]
                cells.append(ms(s.rf_auprc))
                means.append(pd.to_numeric(s.rf_auprc).mean() if len(s) else np.nan)
            if e != "nodp":
                j = int(np.nanargmax(means[:5]))
                cells[j] = r"\textbf{" + cells[j] + "}"
            L.append(f"{LBL[ds] if e == '1' else ''} & {ETX[e]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    write(args.out, "tab_baselines_auprc.tex", L)

    # ---- Table: ablation ------------------------------------------------------------------
    var = [("common", "Common stream only"), ("abl:plain_oversample", r"+ $n_r$ plain samples $G(z,\cmin)$"),
           ("abl:no_judge", "w/o private judge (Stage 4)"),
           ("abl:judge_disc", "judge = DP discriminator (no extra budget)"),
           ("abl:no_maxmin", "w/o diversity selection (Stage 5)"),
           ("abl:no_flow_refine", "w/o flow scoring and refinement (Stages 2--3)"),
           ("abl:no_refine", "w/o latent refinement (Stage 3)"),
           ("abl:judge_rf_nonprivate", r"judge = non-private RF$^\ddagger$"),
           ("combined", r"Full sampler (\method{})")]
    full = ours[ours.config == "combined"].set_index(key).rf_auprc.astype(float)
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Ablation of the rare-minority sampler. Left: RF AUPRC at $\varepsilon=3$ (mean $\pm$ std over 5 seeds). "
         r"Right: difference to the full sampler, number of runs in which the variant is worse, and Wilcoxon "
         r"signed-rank $p$-value, over the 60 DP runs (4 datasets $\times$ 3 budgets $\times$ 5 seeds) and over all 80 "
         r"runs (including No-DP). Every variant adds the same number $n_r$ of minority records to the same common "
         r"stream. On Credit Card Fraud at $\varepsilon \le 3$ the private judge accepts almost no candidate, so the full "
         r"sampler and the judge-based variants consist of plain samples there (Section~\ref{sec:res-failure}). "
         r"$^\ddagger$Reference only: trained on real data outside the privacy budget.}",
         r"\label{tab:ablation}", r"\small", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccccccccc}", r"\toprule",
         r" & \multicolumn{4}{c}{RF AUPRC, $\varepsilon=3$} & \multicolumn{3}{c}{DP runs (60), vs.\ full sampler} "
         r"& \multicolumn{3}{c}{All runs (80), vs.\ full sampler} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
         r"Variant & Adult & Taiwan & Credit Card Fraud & MIMIC-III & $\Delta$ AUPRC & worse & $p$ "
         r"& $\Delta$ AUPRC & worse & $p$ \\", r"\midrule"]
    a3 = ours[ours.eps == "3"]
    for cfg, txt in var:
        cells = [ms(a3[(a3.dataset == ds) & (a3.config == cfg)].rf_auprc) for ds in DS]
        if cfg == "combined":
            tail = ["--"] * 6
            L.append(r"\midrule")
        else:
            x = ours[ours.config == cfg].set_index(key).rf_auprc.astype(float)
            dd = (x - full).dropna()
            dp = dd[dd.index.get_level_values("eps") != "nodp"]
            tail = []
            for z in (dp, dd):
                tail += [f"{z.mean():+.4f}", f"{int((z < 0).sum())}/{len(z)}", pfmt(wilcoxon(z).pvalue)]
        L.append(txt + " & " + " & ".join(cells + tail) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    write(args.out, "tab_ablation.tex", L)

    # ---- Table: privacy budget per component ----------------------------------------------
    c = ours[(ours.config == "combined") & (ours.eps != "nodp")]
    L = [r"\begin{tabular}{llcccccccc}", r"\toprule",
         r"Dataset & $\varepsilon$ & $\epsD$ & $\epsS$ & $\epsJ$ & Total (basic) & Total (RDP) & "
         r"$\sigma_S/\nhat$ & $w_{\mathrm{MM}}$ & $\sigma_J$ \\", r"\midrule"]
    for ds in DS:
        for e in ["10", "3", "1"]:
            x = c[(c.dataset == ds) & (c.eps == e)].mean(numeric_only=True)
            L.append(f"{LBL[ds]} & {e} & {x.budget_epsilon_gan_achieved:.2f} & {x.budget_epsilon_stats:.2f} & "
                     f"{x.budget_epsilon_judge_achieved:.2f} & {x.budget_epsilon_total_basic:.2f} & "
                     f"{x.budget_epsilon_total_rdp:.2f} & {x.budget_stats_rel_noise:.3f} & {x.budget_mm_weight:.2f} & "
                     f"{x.budget_judge_noise_multiplier:.2f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    cap = (r"Privacy budget per component (mean over seeds): achieved $\varepsilon$ of the discriminator ($\epsD$), the "
           r"minority statistics ($\epsS$) and the judge ($\epsJ$), each converted at $\delta/3$; total by basic "
           r"composition (must not exceed the target) and by RDP composition of the same three mechanisms; relative noise "
           r"of the released minority sums ($\sigma_S/\nhat$); moment-matching weight $w_{\mathrm{MM}}$; and the judge's "
           r"noise multiplier $\sigma_J$. The RDP total adds the RDP curves of the three mechanisms (the statistics as a "
           r"single Gaussian mechanism with noise multiplier $\sigma_S/\Delta_2$) and converts once at $\delta$ instead of "
           r"$\delta/3$, which is why it can be smaller than $\epsD$ alone. Totals are computed before rounding.")
    write(args.out, "tab_budget.tex", wrap("\n".join(L), cap, "tab:budget", "t", wide=True))

    # ---- Table: DP accounting details (discriminator, judge, statistics) ------------------
    L = [r"\begin{tabular}{llccccccccccc}", r"\toprule",
         r" & & \multicolumn{5}{c}{Discriminator (DP-SGD)} & \multicolumn{3}{c}{Judge (DP-SGD)} & "
         r"\multicolumn{2}{c}{Statistics} & \\",
         r"\cmidrule(lr){3-7}\cmidrule(lr){8-10}\cmidrule(lr){11-12}",
         r"Dataset & $\varepsilon$ & $C$ & $\sigma$ & $q$ & $T$ & $\epsD$ & $\sigma_J$ & $q_J$ & $T_J$ & $\Delta_2$ & "
         r"$\sigma_S$ & $\delta$ \\", r"\midrule"]
    for ds in DS:
        for e in ["10", "3", "1"]:
            m = run_meta(args.root, ds, e)
            L.append(f"{LBL[ds]} & {e} & {m.C:.2f} & {m.sig_d:.3f} & {m.q_d:.4f} & {m.T_d:.0f} & {m.eps_d:.3f} & "
                     f"{m.sig_j:.2f} & {m.q_j:.4f} & {m.T_j:.0f} & {m.delta2:.2f} & {m.sig_s:.1f} & {m.delta:.1e}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    cap = (r"DP accounting (identical for all seeds): clipping norm $C$, noise multiplier $\sigma$, Poisson sampling rate "
           r"$q$, number of steps $T$ and achieved $\epsD$ (at $\delta/3$) of the discriminator; noise multiplier, sampling "
           r"rate and steps of the judge ($C = 1$, 10 epochs); $\ell_2$ sensitivity $\Delta_2 = \sqrt{2}\,B$ and noise "
           r"standard deviation $\sigma_S$ of the minority statistics; total $\delta = \min(10^{-5}, 1/(10\,|\Dtr|))$. "
           r"The budget shares are in Table~\ref{tab:budget}.")
    write(args.out, "tab_dp.tex", wrap("\n".join(L), cap, "tab:dpsgd", "htbp", wide=True))

    # ---- Table: selected hyperparameters ------------------------------------------------
    L = [r"\begin{tabular}{llccccc}", r"\toprule",
         r"Dataset & Setting & Epochs & Batch & Generator lr & Discriminator lr & $C$ / $\gamma$ \\", r"\midrule"]
    for ds in DS:
        for e in ["nodp", "1", "3", "10"]:
            bp = json.load(open(R(args.hparams, f"{ds}_eps{e}.json")))["best_params"]
            v = {**PRESET_ALL, **PRESET[ds], **bp}
            reg = f"$\\gamma = {v['r1_gamma']:.2f}$" if e == "nodp" else f"$C = {v['dp_max_grad_norm']:.2f}$"
            L.append(f"{LBL[ds] if e == 'nodp' else ''} & {ETX[e]} & {v['gan_epochs']} & {v['batch_size']} & "
                     f"{v['gen_lr']:.2e} & {v['disc_lr']:.2e} & {reg}" + r" \\")
        L.append(r"\midrule" if ds != DS[-1] else "")
    L = [x for x in L if x] + [r"\bottomrule", r"\end{tabular}"]
    cap = (r"Hyperparameters selected by the search of Appendix~\ref{app:tuning} (the upper epoch bound of the search is "
           r"the dataset preset). Presets used for the untuned runs: epochs 80 (Adult, Taiwan), 50 (Credit Card Fraud), "
           r"100 (MIMIC-III); batch size 512; both learning rates $2 \times 10^{-4}$; $C = 5$ (Adult, Taiwan) or 1; "
           r"$\gamma = 5$ (Adult) or 10. Identical rows (e.g.\ Taiwan at $\varepsilon = 1$ and 3) arise because every "
           r"search used the same seeded sampler on the same space, so its random start-up trials coincide.")
    write(args.out, "tab_tuned.tex", wrap("\n".join(L), cap, "tab:tuned", "htbp"))

    # ---- Table: tuned vs untuned --------------------------------------------------------
    L = [r"\begin{tabular}{llcccccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{Tuned hyperparameters} & \multicolumn{3}{c}{Dataset presets (untuned)} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Dataset & Setting & Common & Combined & Gain & Common & Combined & Gain \\", r"\midrule"]
    for ds in DS:
        for e in ["nodp", "10", "3", "1"]:
            cells = []
            for src in (ours, uo):
                x = src[(src.dataset == ds) & (src.eps == e)]
                a = x[x.config == "common"].set_index("seed").rf_auprc.astype(float)
                b = x[x.config == "combined"].set_index("seed").rf_auprc.astype(float)
                cells += [ms(a), ms(b), f"{(b - a).mean():+.3f}"]
            L.append(f"{LBL[ds]} & {ETX[e]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    cap = (r"RF AUPRC with the tuned hyperparameters (Table~\ref{tab:tuned}) and with the dataset presets, i.e.\ "
           r"without any tuning (mean $\pm$ std over 5 seeds; same splits, sampler and evaluation). Gains are computed "
           r"from unrounded means.")
    write(args.out, "tab_untuned.tex", wrap("\n".join(L), cap, "tab:untuned", "htbp", wide=True))

    # ---- Tables copied from aggregate.py with manuscript captions -------------------------
    def load(name):
        return open(os.path.join(SRC, name)).read().replace(", ours", r", \method{}").replace("privatised", "privatized")

    t = set_caption(load("tab_utility.tex"),
                    r"TSTR utility on the sealed test sets (mean $\pm$ std over 5 seeds). Common: conditional sampling "
                    r"only; combined: common plus rare-minority stream; TRTR: the same classifiers trained on real "
                    r"training data. F1 at the threshold that maximizes F1 on the real validation split (an oracle "
                    r"calibration, Section~\ref{sec:metrics}). Bold: better of common and combined.")
    write(args.out, "tab_utility.tex", [t.rstrip("\n")])
    t = set_caption(load("tab_fidelity_privacy.tex"),
                    r"Fidelity (lower is better; KS (minority) compares the minority class only) and membership inference "
                    r"against the released data (attack AUC 0.5 and DCR share 0.5 indicate no measurable leakage); "
                    r"mean $\pm$ std over 5 seeds.")
    write(args.out, "tab_fidelity_privacy.tex", [t.rstrip("\n")])
    t = set_caption(load("tab_gate.tex"),
                    r"PrivacyGate audit: fraction of synthetic records that the gate flags as near-copies of training "
                    r"records or as outliers (mean over seeds; fractions may not sum to 1 because of rounding). "
                    r"\method{} does not filter; the gate only audits.")
    write(args.out, "tab_gate.tex", [t.rstrip("\n")])
    t = set_caption(load("tab_baselines.tex"),
                    r"Full comparison with the baselines on identical splits and evaluation (mean $\pm$ std over seeds; "
                    r"\method{}, matched SmartNoise generators and SMOTE 5 seeds; SmartNoise generators with default "
                    r"settings, CTGAN and TVAE 3 seeds). (SmartNoise): library defaults, i.e.\ the library's $\delta$ "
                    r"($1/(n\sqrt{n})$ for DP-CTGAN and PATE-CTGAN, $10^{-9}$ for MST) and 10\% of $\varepsilon$ spent on "
                    r"private column bounds; (SmartNoise, matched): the same $\delta$ and public column bounds as "
                    r"\method{}, the full $\varepsilon$ for the generator. $^\dagger$Not differentially private (listed "
                    r"under No-DP).", label="tab:baselines-full")
    write(args.out, "tab_baselines_full.tex", [t.rstrip("\n")])

    body = lambda name: re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", load(name), flags=re.S).group(0)
    write(args.out, "tab_baseline_minority.tex", wrap(
        body("tab_baseline_minority.tex"),
        r"Fraction of minority records in the synthetic data of each baseline (mean over seeds). Real minority rates: "
        r"Adult 0.248, Taiwan 0.221, Credit Card Fraud 0.0049, MIMIC-III 0.113.", "tab:baseline-minority", wide=True))
    write(args.out, "tab_confusion.tex", wrap(
        body("tab_confusion.tex"),
        r"Random-forest confusion matrices on the sealed test sets at the threshold selected on the real validation "
        r"split (an oracle calibration, Section~\ref{sec:metrics}). Counts are rounded means over 5 seeds; precision, "
        r"recall and F1 are means $\pm$ std of the per-seed values, not ratios of the mean counts.", "tab:cm", wide=True))
    t = body("tab_nonprivate.tex").replace("{RareSel-DP}", r"{\method{}}")
    write(args.out, "tab_nonprivate_variant.tex", wrap(
        t, r"RF AUPRC of \method{} and of a non-private variant of the pipeline, an upper reference that uses real data "
           r"outside the budget (exact minority statistics, feature and correlation matching on real batches, a "
           r"random-forest judge trained on real data, quantile matching, and a filtering PrivacyGate; discriminator "
           r"trained with DP-SGD). Mean $\pm$ std over 5 seeds; same splits, hyperparameters and evaluation. Gains "
           r"are computed from unrounded means.", "tab:nonprivate", wide=True))

    for f in sorted(os.listdir(args.out)):
        print(os.path.join(args.out, f))


if __name__ == "__main__":
    main()
