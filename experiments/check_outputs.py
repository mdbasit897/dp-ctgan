#!/usr/bin/env python3
"""Check that every input and output the manuscript needs is present and complete.

  python3 experiments/check_outputs.py                       # from the repository root
  python3 experiments/check_outputs.py --baseline-seeds 5    # if the baselines use all 5 seeds

Checks: prepared datasets, tuned hyperparameters, the 80-run grid (completeness, tuned settings
actually used, achieved epsilon, ablations, PrivacyGate), the seed-0 artifacts the figures need,
baselines, and the aggregated tables / figures. Prints a short results overview at the end.
"""
import argparse
import glob
import json
import os

import numpy as np

DATASETS = ["adult", "taiwan_credit", "credit_fraud", "mimic3"]
EPS = ["nodp", "1", "3", "10"]
SEEDS = range(5)
DATA = {"adult": "adult.csv", "taiwan_credit": "UCI_Credit_Card.csv", "credit_fraud": "creditcard.csv",
        "mimic3": "mimic3.csv"}
ABLATIONS = {"plain_oversample", "no_qm", "no_flow_refine", "no_refine", "no_judge", "no_maxmin"}
ARTIFACTS = ["synthetic_common.csv.gz", "synthetic_rare.csv.gz", "probs.npz", "mia_scores.npz", "sampler_diag.npz"]
TABLES = ["tab_utility.tex", "tab_fidelity_privacy.tex", "tab_ablation.tex", "tab_baselines.tex", "tab_gate.tex",
          "tab_dp.tex", "tab_confusion.tex", "stats.txt", "runs.csv", "mean_std.csv"]
FIGURES = ["fig_privacy_utility", "fig_gains", "fig_ablation", "fig_pr_curves", "fig_mia_roc_domias", "fig_sampler_scores"] + \
          [f"fig_{k}_{d}" for k in ("marginals", "embedding", "corrdiff") for d in DATASETS]
BASELINES = {"smote": ["nodp"], "ctgan": ["nodp"], "tvae": ["nodp"], "pategan": ["1", "3", "10"],
             "dpgan": ["1", "3", "10"], "aim": ["1", "3", "10"]}

ap = argparse.ArgumentParser()
ap.add_argument("--baseline-seeds", type=int, default=3, help="number of seeds the baselines are run with")
BL_SEEDS = ap.parse_args().baseline_seeds

problems, warnings = [], []


def section(t):
    print(f"\n== {t} " + "=" * max(0, 70 - len(t)))


def ok(cond, msg_ok, msg_bad, warn=False):
    print(("  [ok]   " if cond else ("  [warn] " if warn else "  [FAIL] ")) + (msg_ok if cond else msg_bad))
    if not cond:
        (warnings if warn else problems).append(msg_bad)


section("1. Datasets (data/)")
for d, f in DATA.items():
    p = os.path.join("data", f)
    ok(os.path.exists(p), f"{p} ({os.path.getsize(p) / 1e6:.1f} MB)" if os.path.exists(p) else "", f"{p} missing")

section("2. Tuned hyperparameters (results/hparams/)")
hp = {}
for d in DATASETS:
    for e in EPS:
        p = f"results/hparams/{d}_eps{e}.json"
        if os.path.exists(p):
            hp[(d, e)] = json.load(open(p))["best_params"]
missing = [f"{d}_eps{e}" for d in DATASETS for e in EPS if (d, e) not in hp]
ok(not missing, "16/16 files", f"missing tuned settings: {missing}")

section("3. Main grid (results/<dataset>/eps<e>/seed<s>/metrics.json)")
runs, bad_runs = {}, []
for d in DATASETS:
    for e in EPS:
        for s in SEEDS:
            run = f"results/{d}/eps{e}/seed{s}"
            if os.path.exists(f"{run}/metrics.json"):
                try:
                    runs[(d, e, s)] = json.load(open(f"{run}/metrics.json"))
                except Exception as ex:
                    bad_runs.append(f"{run}: unreadable ({ex})")
            if os.path.exists(f"{run}/error.txt"):
                warnings.append(f"{run}/error.txt exists (stale if metrics.json is there; delete it)")
                print(f"  [warn] {run}/error.txt exists")
ok(len(runs) == 80, "80/80 runs", f"only {len(runs)}/80 runs have metrics.json")
for d in DATASETS:
    n = sum(1 for k in runs if k[0] == d)
    print(f"         {d:14s} {n}/20")

untuned, quick, no_abl, no_gate, eps_over = [], [], [], [], []
for (d, e, s), R in runs.items():
    tag = f"{d}/eps{e}/seed{s}"
    m = R["meta"]
    if m.get("quick"):
        quick.append(tag)
    for k, v in hp.get((d, e), {}).items():
        got = m["hp"].get(k)
        if got is None or (isinstance(v, (int, float)) and not np.isclose(float(got), float(v))) or \
                (not isinstance(v, (int, float)) and got != v):
            untuned.append(tag)
            break
    if set(R.get("ablations", {})) != ABLATIONS:
        no_abl.append(tag)
    if not all((R.get("privacy_gate") or {}).get(k) for k in ("common", "rare")):
        no_gate.append(tag)
    if e != "nodp" and (m.get("achieved_epsilon") is None or m["achieved_epsilon"] > float(e) + 1e-3):
        eps_over.append(f"{tag} (achieved {m.get('achieved_epsilon')})")
    for key in ("trtr", "common", "combined"):
        if key not in R:
            bad_runs.append(f"{tag}: no '{key}' block")
ok(not bad_runs, "every run has TRTR, common and combined results", f"incomplete runs: {bad_runs}")
ok(not quick, "no smoke-test (--quick) runs mixed in", f"--quick runs in results/: {quick}")
ok(not untuned, "every run used its tuned hyperparameters", f"runs NOT using tuned settings: {untuned}")
ok(not no_abl, "all 6 ablation variants in every run", f"runs missing ablations: {no_abl}")
ok(not no_gate, "PrivacyGate statistics in every run", f"runs missing PrivacyGate stats: {no_gate}", warn=True)
ok(not eps_over, "achieved epsilon <= target in every DP run", f"achieved epsilon above target: {eps_over}")

section("4. Seed-0 artifacts for the figures")
miss = [f"{d}/eps{e}/seed0/{a}" for d in DATASETS for e in EPS for a in ARTIFACTS
        if not os.path.exists(f"results/{d}/eps{e}/seed0/{a}")]
ok(not miss, f"all {len(DATASETS) * len(EPS) * len(ARTIFACTS)} artifact files present",
   f"{len(miss)} artifact files missing, e.g. {miss[:4]}")

section("5. Baselines (results/baselines/)")
for meth, eps_l in BASELINES.items():
    want = len(DATASETS) * len(eps_l) * BL_SEEDS
    files = [f for s in range(BL_SEEDS) for f in glob.glob(f"results/baselines/{meth}/*/eps*/seed{s}/metrics.json")]
    errs = glob.glob(f"results/baselines/{meth}/*/eps*/seed*/error.txt")
    ok(len(files) == want and not errs, f"{meth:8s} {len(files)}/{want}",
       f"{meth:8s} {len(files)}/{want} done, {len(errs)} errors" + (f" (e.g. {errs[0]})" if errs else ""), warn=True)
    if meth in ("ctgan", "tvae", "dpgan"):
        old_runs = [f for f in files if not json.load(open(f))["meta"].get("n_iter")]
        ok(not old_runs, f"{meth:8s} all runs use the fixed --n-iter setting",
           f"{meth:8s} {len(old_runs)} run(s) made with synthcity's default length; delete them: {old_runs[:3]}")

section("6. Aggregated tables (results/summary/) and figures (results/figures/)")
miss_t = [t for t in TABLES if not os.path.exists(f"results/summary/{t}")]
ok(not miss_t, "all tables present", f"missing tables: {miss_t} -> run experiments/aggregate.py", warn=True)
miss_f = [f for f in FIGURES if not os.path.exists(f"results/figures/{f}.pdf")]
ok(not miss_f, f"all {len(FIGURES)} figures present", f"missing figures: {miss_f} -> run experiments/make_figures.py",
   warn=True)
if os.path.exists("results/summary/runs.csv") and runs:
    newest_run = max(os.path.getmtime(f"results/{d}/eps{e}/seed{s}/metrics.json") for d, e, s in runs)
    ok(os.path.getmtime("results/summary/runs.csv") >= newest_run, "tables are newer than the last run",
       "tables are older than some runs -> rerun aggregate.py and make_figures.py", warn=True)

section("7. Results overview: RF AUPRC on the test set (mean over seeds)")
print(f"  {'dataset':14s} {'budget':6s} {'TRTR':>7s} {'common':>7s} {'combined':>9s} {'diff':>7s}  "
      f"{'DOMIAS AUC':>10s} {'eps achieved':>12s}")
for d in DATASETS:
    for e in EPS:
        R = [runs[(d, e, s)] for s in SEEDS if (d, e, s) in runs]
        if not R:
            continue
        g = lambda f: np.mean([f(r) for r in R])
        tr = g(lambda r: r["trtr"]["rf"]["auprc"])
        co = g(lambda r: r["common"]["utility"]["rf"]["auprc"])
        cb = g(lambda r: r["combined"]["utility"]["rf"]["auprc"])
        dm = g(lambda r: r["combined"]["privacy"]["domias"]["auc"])
        ae = "-" if e == "nodp" else f"{g(lambda r: r['meta']['achieved_epsilon']):.3f}"
        print(f"  {d:14s} {e:6s} {tr:7.3f} {co:7.3f} {cb:9.3f} {cb - co:+7.3f}  {dm:10.3f} {ae:>12s}")

section("Summary")
if problems:
    print(f"  {len(problems)} problem(s) to fix before writing up:")
    for p in problems:
        print("   -", p)
else:
    print("  Main grid complete and consistent.")
if warnings:
    print(f"  {len(warnings)} item(s) still to do or check:")
    for w in warnings:
        print("   -", w)
