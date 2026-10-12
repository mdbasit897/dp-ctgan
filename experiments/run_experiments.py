#!/usr/bin/env python3
"""Run the full experiment grid: datasets x privacy budgets x seeds.

For every (dataset, epsilon, seed) it writes results/<dataset>/eps<e>/seed<s>/metrics.json
containing TRTR, common, combined and (optionally) ablation results, PrivacyGate
statistics, DP accounting, and timings. Runs are skipped if metrics.json already
exists, so the script can be restarted after a VM preemption.

Examples
  # smoke test (minutes, tiny settings)
  python experiments/run_experiments.py --datasets adult --eps 3 --seeds 0 --quick
  # full grid, split across 3 worker processes (run each in its own tmux window)
  python experiments/run_experiments.py --shard 0/3 --ablations
  python experiments/run_experiments.py --shard 1/3 --ablations
  python experiments/run_experiments.py --shard 2/3 --ablations
"""
import argparse
import json
import os
import time
import traceback
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import (DATASETS, P, eps_tag, fidelity, load, make_splits, parse_eps, privacy, tstr)

ABLATIONS = {
    # name: (sample_rare flags, apply quantile matching)
    "no_qm": (dict(), False),
    "no_flow_refine": (dict(use_flow=False, use_refine=False), True),
    "no_refine": (dict(use_refine=False), True),
    "no_judge": (dict(use_judge=False), True),
    "no_maxmin": (dict(use_maxmin=False), True),
}


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def quick_overrides(hp):
    hp.gan_epochs, hp.flow_epochs, hp.finetune_epochs = 3, 3, 2
    hp.rare_pool_size, hp.rare_max_iters, hp.refine_steps, hp.flow_n_samples = 2000, 2, 5, 1000
    return hp


def make_rare(model, n_rare, splits, target, cat_cols, cont_cols, feat, cfg, hp, flags, use_qm, diag=False):
    train = splits["train"]
    real_min = train[train[target] == 1].reset_index(drop=True)
    res = model.sample_rare(n_rare, df_real=train, hp=hp, return_diagnostics=diag, **flags)
    rare, d = res if diag else (res, None)
    if use_qm:
        rare = P.quantile_match(rare, real_min, cont_cols, blend=hp.quantile_blend, orig_df=train)
    rare[target] = 1
    rare = P.postprocess_synthetic(rare, train, target, cat_cols, cfg.int_cols, cfg.hard_bounds)
    stats = None
    if hp.privacy_filter_enabled:
        rare, stats = P.privacy_realism_filter(rare, real_min, feat, desired_n=n_rare, label="rare",
                                               max_ref=hp.privacy_max_ref, return_stats=True, verbose=False)
    return rare, stats, d


def run_one(dataset, eps, seed, args):
    t_all = time.time()
    out_dir = os.path.join(args.out, dataset, f"eps{eps_tag(eps)}", f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    P.set_seed(seed)
    cfg, df, target, cat_cols = load(dataset, args.data_dir, verbose=True)
    feat = [c for c in df.columns if c != target]
    cont_cols = [c for c in feat if c not in cat_cols]
    splits = make_splits(df, target, seed)
    train = splits["train"]

    overrides = {}
    hp_file = os.path.join(args.hp_dir, f"{dataset}_eps{eps_tag(eps)}.json")
    if os.path.exists(hp_file):
        overrides = json.load(open(hp_file))["best_params"]
        print(f"  using tuned hyperparameters from {hp_file}: {overrides}")
    hp = P.hyperparams_for(dataset, overrides)
    if args.quick:
        hp = quick_overrides(hp)
    if args.delta is not None:
        hp.delta = args.delta

    scaler = StandardScaler().fit(P._ensure_numeric(train, target)[feat].values.astype(float))
    timings, R = {}, {"meta": {
        "dataset": dataset, "epsilon": eps, "seed": seed, "n_total": len(df),
        "n_train": len(train), "n_val": len(splits["val"]), "n_holdout": len(splits["holdout"]),
        "n_test": len(splits["test"]), "minority_rate_test": float(splits["test"][target].mean()),
        "n_features": len(feat), "n_categorical": len(cat_cols), "hp": asdict(hp), "quick": args.quick}}
    save_art = seed == args.artifact_seed

    t = time.time()
    R["trtr"], probs_trtr = tstr(train, splits, scaler, feat, target, hp.knn_k, seed, return_probs=True)
    timings["trtr"] = time.time() - t

    t = time.time()
    model = P.DPCTGAN(epsilon=eps, hp=hp, verbose=True)
    model.fit(train, cat_cols, target, ds_name=dataset)
    timings["gan_train"] = time.time() - t
    R["meta"]["dp"] = model.dp_info
    R["meta"]["achieved_epsilon"] = None if eps is None else float(model._actual_epsilon)

    t = time.time()
    common = model.sample_common(len(train))
    common = P.postprocess_synthetic(common, train, target, cat_cols, cfg.int_cols, cfg.hard_bounds)
    gate = {}
    if hp.privacy_filter_enabled:
        common, gate["common"] = P.privacy_realism_filter(common, train, feat, desired_n=len(train), label="common",
                                                          max_ref=hp.privacy_max_ref, return_stats=True,
                                                          verbose=False)
    timings["sample_common"] = time.time() - t

    t = time.time()
    model.build_flows(hp)
    timings["flows"] = time.time() - t
    n_rare = max(200, int(train[target].sum() * hp.rare_multiplier))
    t = time.time()
    rare, gate["rare"], diag = make_rare(model, n_rare, splits, target, cat_cols, cont_cols, feat, cfg, hp,
                                         {}, True, diag=True)
    timings["sample_rare"] = time.time() - t
    combined = pd.concat([common, rare], ignore_index=True)[df.columns]
    R["meta"]["n_rare"] = int(len(rare))
    R["privacy_gate"] = gate

    probs = {"y_test": splits["test"][target].values.astype(np.int8),
             **{f"trtr_{k}": v for k, v in probs_trtr.items()}}
    mia_scores = {}
    for name, synth in (("common", common), ("combined", combined)):
        t = time.time()
        u, pr = tstr(synth, splits, scaler, feat, target, hp.knn_k, seed, return_probs=True)
        R[name] = {"utility": u, "fidelity": fidelity(train, synth, feat, target, seed)}
        R[name]["privacy"], sc = privacy(splits, synth, feat, target, seed, return_scores=True)
        probs.update({f"{name}_{k}": v for k, v in pr.items()})
        mia_scores.update({f"{name}_{k}": v for k, v in sc.items()})
        timings[f"eval_{name}"] = time.time() - t

    if args.ablations:
        R["ablations"] = {}
        t = time.time()
        plain = model.sample_minority_plain(n_rare)
        plain = P.postprocess_synthetic(plain, train, target, cat_cols, cfg.int_cols, cfg.hard_bounds)
        if hp.privacy_filter_enabled:
            plain = P.privacy_realism_filter(plain, train[train[target] == 1], feat, desired_n=n_rare,
                                             max_ref=hp.privacy_max_ref, verbose=False)
        variants = {"plain_oversample": plain}
        for name, (flags, use_qm) in ABLATIONS.items():
            variants[name], _, _ = make_rare(model, n_rare, splits, target, cat_cols, cont_cols, feat, cfg, hp,
                                             flags, use_qm)
        for name, rv in variants.items():
            comb_v = pd.concat([common, rv], ignore_index=True)[df.columns]
            R["ablations"][name] = {"utility": tstr(comb_v, splits, scaler, feat, target, hp.knn_k, seed),
                                    "fidelity": dict(zip(("ks", "js"), P.compute_ks_js(train, comb_v, feat))),
                                    "n_rare": int(len(rv))}
        timings["ablations"] = time.time() - t

    timings["total"] = time.time() - t_all
    R["timings_sec"] = timings

    if save_art:
        common.to_csv(os.path.join(out_dir, "synthetic_common.csv.gz"), index=False)
        rare.to_csv(os.path.join(out_dir, "synthetic_rare.csv.gz"), index=False)
        np.savez_compressed(os.path.join(out_dir, "probs.npz"), **probs)
        np.savez_compressed(os.path.join(out_dir, "mia_scores.npz"), **mia_scores)
        if diag:
            np.savez_compressed(os.path.join(out_dir, "sampler_diag.npz"),
                                **{k: v for k, v in diag.items() if len(v)})

    tmp = os.path.join(out_dir, "metrics.json.tmp")
    with open(tmp, "w") as f:
        json.dump(R, f, indent=1, default=_json_default)
    os.replace(tmp, os.path.join(out_dir, "metrics.json"))
    print(f"  saved {out_dir}/metrics.json  ({timings['total']/60:.1f} min)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--eps", nargs="+", default=["none", "1", "3", "10"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--ablations", action="store_true", help="also run the sampler ablations (slower)")
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    ap.add_argument("--shard", default="0/1", help="i/N: run every N-th job starting at i")
    ap.add_argument("--data-dir", default=None, help="directory with the prepared CSVs (default: ./data)")
    ap.add_argument("--hp-dir", default="results/hparams", help="tuned hyperparameters from tune.py")
    ap.add_argument("--out", default="results")
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--artifact-seed", type=int, default=0, help="seed whose synthetic data/scores are saved for figures")
    args = ap.parse_args()

    jobs = [(d, parse_eps(e), s) for d in args.datasets for e in args.eps for s in args.seeds]
    i, n = map(int, args.shard.split("/"))
    jobs = jobs[i::n]
    print(f"{len(jobs)} jobs in shard {args.shard}")
    for d, e, s in jobs:
        done = os.path.join(args.out, d, f"eps{eps_tag(e)}", f"seed{s}", "metrics.json")
        if os.path.exists(done):
            print(f"skip {done}")
            continue
        print(f"\n=== {d}  eps={eps_tag(e)}  seed={s} ===")
        try:
            run_one(d, e, s, args)
        except Exception:
            traceback.print_exc()
            err = os.path.join(args.out, d, f"eps{eps_tag(e)}", f"seed{s}", "error.txt")
            os.makedirs(os.path.dirname(err), exist_ok=True)
            open(err, "w").write(traceback.format_exc())
        finally:
            P._cuda_clear()


if __name__ == "__main__":
    main()
