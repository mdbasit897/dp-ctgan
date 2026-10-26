#!/usr/bin/env python3
"""Run baseline generators on exactly the same splits and evaluation as run_experiments.py.

Methods (install the libraries in SEPARATE virtual environments, see README.md):
  synthcity : pategan, dpgan, aim, privbayes (DP);  ctgan, tvae (non-private references)
  smartnoise: sn_dpctgan, sn_patectgan, sn_mst (DP)
  imblearn  : smote (non-private oversampling of the real training data; utility reference only)

Writes results/baselines/<method>/<dataset>/eps<e>/seed<s>/metrics.json.

  python experiments/run_baselines.py --methods ctgan tvae --eps none --seeds 0 1 2
  python experiments/run_baselines.py --methods smote --eps none

SmartNoise settings. Default ("library defaults"): the library infers the continuous column
bounds privately with preprocessor_eps = 0.1 eps (subtracted from the training budget) and uses its own
delta (1/(n sqrt(n)) for DP-CTGAN and PATE-CTGAN, 1e-9 for MST). With --matched the baselines get the
same treatment as RareSel-DP: delta = min(1e-5, 1/(10 n_train)) and the training-range column bounds
treated as public (as RareSel-DP's encoder and clipping ranges), so preprocessor_eps = 0 and the full
eps goes to the generator. --matched writes to <method>_matched/ so both variants can be compared:

  python experiments/run_baselines.py --methods sn_dpctgan sn_patectgan sn_mst --matched --seeds 0 1 2 3 4

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


def gen_synthcity(name, train, target, eps, delta, n, seed, n_iter=0, extra_kwargs=None):
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader
    kw = {"random_state": seed, **(extra_kwargs or {})}
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


SN_DEFAULT_PREPROCESSOR_FRAC = 0.1


def sn_library_delta(name, n_train):
    """delta that SmartNoise uses when none is passed (snsynth 1.0.x)."""
    return 1e-9 if name == "sn_mst" else 1.0 / (n_train * np.sqrt(n_train))


def gen_smartnoise(name, train, target, cat_cols, eps, n, delta=None, public_bounds=False):
    """delta=None and public_bounds=False give the library defaults."""
    from snsynth import Synthesizer
    cont = [c for c in train.columns if c not in cat_cols + [target]]
    kw = {} if delta is None else {"delta": float(delta)}
    synth = Synthesizer.create(name.replace("sn_", ""), epsilon=float(eps), verbose=False, **kw)
    if public_bounds:
        # column ranges of the training data, treated as public exactly as RareSel-DP's encoder and
        # clipping ranges; no budget is spent on preprocessing. Same column transformers as the library
        # default: min-max scaling (GAN style) or 10 equal-width bins (MST, cube style).
        from snsynth.transform import BinTransformer, MinMaxTransformer
        cons = {}
        for c in cont:
            lo, hi = float(train[c].min()), float(train[c].max())
            hi = hi if hi > lo else lo + 1.0
            cons[c] = (BinTransformer(bins=10, lower=lo, upper=hi) if name == "sn_mst"
                       else MinMaxTransformer(lower=lo, upper=hi, negative=True))
        synth.fit(train, transformer=cons, categorical_columns=cat_cols + [target], continuous_columns=[],
                  preprocessor_eps=0.0)
    else:
        synth.fit(train, categorical_columns=cat_cols + [target], continuous_columns=cont,
                  preprocessor_eps=SN_DEFAULT_PREPROCESSOR_FRAC * float(eps))
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
        synth = gen_smartnoise(method, tr, target, cat_cols, eps, len(tr),
                               delta=delta if args.matched else None, public_bounds=args.matched)
    else:
        synth = gen_synthcity(method, tr, target, eps, delta, len(tr), seed, args.n_iter, args.plugin_kwargs)
    secs = time.time() - t
    synth = synth[df.columns].copy()
    synth[target] = (pd.to_numeric(synth[target], errors="coerce").fillna(0) > 0.5).astype(int)
    for c in feat:
        synth[c] = pd.to_numeric(synth[c], errors="coerce").fillna(tr[c].median())
    synth = P.postprocess_synthetic(synth, tr, target, cat_cols, cfg.int_cols, cfg.hard_bounds)

    scaler = StandardScaler().fit(P._ensure_numeric(tr, target)[feat].values.astype(float))
    # the delta actually used by the generator (with library defaults, SmartNoise uses its own delta)
    if method.startswith("sn_"):
        sn = {"matched": bool(args.matched), "public_bounds": bool(args.matched),
              "preprocessor_eps": 0.0 if args.matched else SN_DEFAULT_PREPROCESSOR_FRAC * float(eps)}
        delta_used = delta if args.matched else sn_library_delta(method, len(tr))
    else:
        sn, delta_used = None, (delta if method in DP_METHODS else None)
    R = {"meta": {"method": args.name or (f"{method}_matched" if args.matched else method), "plugin": method,
                  "plugin_kwargs": args.plugin_kwargs, "dataset": ds, "epsilon": eps, "seed": seed,
                  "delta": delta_used, "smartnoise": sn,
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
    ap.add_argument("--plugin-kwargs", type=json.loads, default=None,
                    help='extra synthcity plugin arguments as JSON, e.g. \'{"lamda": 0.01}\'')
    ap.add_argument("--name", default=None,
                    help="folder/label for the results (default: the method name); one method per call")
    ap.add_argument("--matched", action="store_true",
                    help="SmartNoise only: RareSel-DP's delta and public training-range bounds (preprocessor_eps = 0); "
                         "results go to <method>_matched/")
    args = ap.parse_args()
    if args.name and len(args.methods) != 1:
        ap.error("--name needs exactly one --methods entry")
    if args.matched and not all(m.startswith("sn_") for m in args.methods):
        ap.error("--matched applies to the SmartNoise methods (sn_*) only")
    for m in args.methods:
        eps_list = [parse_eps(e) for e in args.eps] if m in DP_METHODS else [None]
        for ds in args.datasets:
            for e in eps_list:
                for s in args.seeds:
                    label = args.name or (f"{m}_matched" if args.matched else m)
                    d = os.path.join(args.out, label, ds, f"eps{eps_tag(e)}", f"seed{s}")
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
