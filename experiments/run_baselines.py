#!/usr/bin/env python3
"""Run baseline generators on exactly the same splits and evaluation as run_experiments.py.

Methods (install the libraries in SEPARATE virtual environments, see RUNNING_EXPERIMENTS.md):
  synthcity : pategan, dpgan, aim, privbayes (DP);  ctgan, tvae (non-private references)
  smartnoise: sn_dpctgan, sn_patectgan, sn_mst (DP)
  imblearn  : smote (non-private oversampling of the real training data; utility reference only)

Writes results/baselines/<method>/<dataset>/eps<e>/seed<s>/metrics.json.

  python experiments/run_baselines.py --methods pategan dpgan aim ctgan tvae --eps 1 3 10
  python experiments/run_baselines.py --methods smote --eps none

Training length: CTGAN, TVAE and DP-GAN run for --n-iter epochs (default 300, the CTGAN paper's
default; synthcity's own default of 2000 takes many hours per run on Credit Card Fraud). The value
used is stored in metrics.json (meta.n_iter). Pass --n-iter 0 to keep synthcity's defaults.
"""
import argparse
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import DATASETS, P, eps_tag, fidelity, load, make_splits, parse_eps, privacy, tstr

DP_METHODS = {"pategan", "dpgan", "aim", "privbayes", "sn_dpctgan", "sn_patectgan", "sn_mst"}
ITER_METHODS = {"ctgan", "tvae", "dpgan"}  # synthcity plugins whose training length is set by n_iter


def gen_synthcity(name, train, target, eps, delta, n, seed, n_iter=0):
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader
    kw = {"random_state": seed}
    if n_iter and name in ITER_METHODS:
        kw["n_iter"] = n_iter
    if name in DP_METHODS:
        kw.update({"epsilon": eps, "delta": delta})
    try:
        gen = Plugins().get(name, **kw)
    except TypeError:
        kw.pop("delta", None)
        gen = Plugins().get(name, **kw)
    gen.fit(GenericDataLoader(train, target_column=target))
    return gen.generate(count=n).dataframe()


def gen_smartnoise(name, train, target, cat_cols, eps, n):
    from snsynth import Synthesizer
    cont = [c for c in train.columns if c not in cat_cols + [target]]
    synth = Synthesizer.create(name.replace("sn_", ""), epsilon=float(eps), verbose=False)
    synth.fit(train, categorical_columns=cat_cols + [target], continuous_columns=cont,
              preprocessor_eps=0.1 * float(eps))
    return synth.sample(n)


def gen_smote(train, target, seed):
    from imblearn.over_sampling import SMOTE
    X, y = train.drop(columns=[target]), train[target].astype(int)
    Xr, yr = SMOTE(random_state=seed).fit_resample(X, y)
    out = pd.DataFrame(Xr, columns=X.columns)
    out[target] = yr.values
    return out


def run_one(method, ds, eps, seed, args):
    cfg, df, target, cat_cols = load(ds, args.data_dir)
    feat = [c for c in df.columns if c != target]
    sp = make_splits(df, target, seed)
    tr = sp["train"]
    P.set_seed(seed)
    delta = args.delta or min(1e-5, 1.0 / (10 * len(tr)))
    t = time.time()
    if method == "smote":
        synth = gen_smote(tr, target, seed)
    elif method.startswith("sn_"):
        synth = gen_smartnoise(method, tr, target, cat_cols, eps, len(tr))
    else:
        synth = gen_synthcity(method, tr, target, eps, delta, len(tr), seed, args.n_iter)
    secs = time.time() - t
    synth = synth[df.columns].copy()
    synth[target] = (pd.to_numeric(synth[target], errors="coerce").fillna(0) > 0.5).astype(int)
    for c in feat:
        synth[c] = pd.to_numeric(synth[c], errors="coerce").fillna(tr[c].median())
    synth = P.postprocess_synthetic(synth, tr, target, cat_cols, cfg.int_cols, cfg.hard_bounds)

    scaler = StandardScaler().fit(P._ensure_numeric(tr, target)[feat].values.astype(float))
    R = {"meta": {"method": method, "dataset": ds, "epsilon": eps, "seed": seed, "delta": delta,
                  "fit_seconds": secs, "n_synth": len(synth), "minority_rate_synth": float(synth[target].mean()),
                  "n_iter": args.n_iter if method in ITER_METHODS and args.n_iter else None},
         "utility": tstr(synth, sp, scaler, feat, target, 5, seed),
         "fidelity": fidelity(tr, synth, feat, target, seed)}
    if method != "smote":  # SMOTE output contains real records; MIA is not meaningful
        R["privacy"] = privacy(sp, synth, feat, target, seed)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--eps", nargs="+", default=["1", "3", "10"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="results/baselines")
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--n-iter", type=int, default=300,
                    help="training epochs for ctgan / tvae / dpgan (0 = synthcity default)")
    args = ap.parse_args()
    for m in args.methods:
        eps_list = [parse_eps(e) for e in args.eps] if m in DP_METHODS else [None]
        for ds in args.datasets:
            for e in eps_list:
                for s in args.seeds:
                    d = os.path.join(args.out, m, ds, f"eps{eps_tag(e)}", f"seed{s}")
                    if os.path.exists(os.path.join(d, "metrics.json")):
                        continue
                    os.makedirs(d, exist_ok=True)
                    print(f"=== {m} {ds} eps={eps_tag(e)} seed={s}")
                    try:
                        R = run_one(m, ds, e, s, args)
                        json.dump(R, open(os.path.join(d, "metrics.json"), "w"), indent=1,
                                  default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
                    except Exception:
                        traceback.print_exc()
                        open(os.path.join(d, "error.txt"), "w").write(traceback.format_exc())


if __name__ == "__main__":
    main()
