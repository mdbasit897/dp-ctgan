#!/usr/bin/env python3
"""Tune GAN hyperparameters with Optuna for each (dataset, epsilon) on seed 0.

Uses only the seed-0 GAN training split (inner 85/15 train / tuning-MIA holdout) and the
validation split, never the test split or the final MIA holdout. Writes
results/hparams/<dataset>_eps<e>.json, which run_experiments.py picks up automatically.

The paper's hyperparameters were selected in --mode nonprivate (the non-private proxy
configuration described in the appendix) and reused for RareSel-DP. Tuning on private data
consumes privacy budget that is NOT included in the reported epsilon (Papernot & Steinke, ICLR 2022).

  python experiments/tune.py --datasets adult taiwan_credit --eps none 3 10 --trials 20
"""
import argparse
import json
import os
import time
from dataclasses import asdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import DATASETS, P, eps_tag, load, make_splits, parse_eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--eps", nargs="+", default=["none", "1", "3", "10"])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=None, help="default: results/hparams (nonprivate) or results_dp/hparams (dp)")
    ap.add_argument("--mode", choices=["nonprivate", "dp"], default="nonprivate",
                    help="privacy mode the GAN is tuned in (run_experiments.py reads --hp-dir)")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.out is None:
        args.out = "results_dp/hparams" if args.mode == "dp" else "results/hparams"
    os.makedirs(args.out, exist_ok=True)

    for ds in args.datasets:
        cfg, df, target, cat_cols = load(ds, args.data_dir, verbose=True)
        feat = [c for c in df.columns if c != target]
        sp = make_splits(df, target, seed=0)
        tr = sp["train"]
        i_g, i_m = train_test_split(range(len(tr)), test_size=0.15, random_state=0, stratify=tr[target])
        gan_tr, mia_ho = tr.iloc[list(i_g)].reset_index(drop=True), tr.iloc[list(i_m)].reset_index(drop=True)
        sc = StandardScaler().fit(P._ensure_numeric(gan_tr, target)[feat].values.astype(float))
        X_val = sc.transform(P._ensure_numeric(sp["val"], target)[feat].values.astype(float))
        y_val = sp["val"][target].values.astype(int)
        for e in map(parse_eps, args.eps):
            path = os.path.join(args.out, f"{ds}_eps{eps_tag(e)}.json")
            if os.path.exists(path):
                print(f"skip {path}")
                continue
            P.set_seed(0)
            hp = P.apply_privacy_mode(P.hyperparams_for(ds), args.mode)
            if args.quick:
                hp.gan_epochs = 3
            t = time.time()
            best = P.optuna_tune(e, gan_tr, sp["val"], mia_ho, sc, X_val, y_val, target, feat, cat_cols,
                                 hp, n_trials=args.trials, seed=0)
            base = asdict(P.apply_privacy_mode(P.hyperparams_for(ds), args.mode))
            changed = {k: v for k, v in asdict(best).items() if base.get(k) != v}
            json.dump({"dataset": ds, "epsilon": e, "trials": args.trials, "best_params": changed,
                       "minutes": (time.time() - t) / 60}, open(path, "w"), indent=1)
            print(f"  wrote {path}: {changed}")


if __name__ == "__main__":
    main()
