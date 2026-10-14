#!/usr/bin/env python3
"""Find PATE-GAN settings under which training does not stop after its first iteration.

In the first baseline round, synthcity's PATE-GAN produced bit-identical output at epsilon = 1 and
epsilon = 3 for every seed: its privacy accountant stops training once the budget is spent, and with
the default settings the very first iteration already exceeds both budgets. This script

  1. prints the PATE-GAN constructor signature of the installed synthcity version, and
  2. fits PATE-GAN on the seed-0 training split at two budgets for each candidate setting and
     reports whether the outputs differ, the fit time, and a quick utility score.

A usable setting gives (a) different outputs at the two budgets and (b) a longer fit at the larger
budget. Re-run the baseline with it, e.g.

  python experiments/run_baselines.py --methods pategan --name pategan --eps 1 3 10 --seeds 0 1 2 \\
      --plugin-kwargs '{"lamda": 0.01}' --out results/baselines

  python experiments/probe_pategan.py                                  # default grid on Adult
  python experiments/probe_pategan.py --grid '{"lamda": [0.001, 0.01, 0.1], "n_teachers": [10, 50]}'
"""
import argparse
import hashlib
import inspect
import itertools
import json
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from common import P, load, make_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="adult")
    ap.add_argument("--eps", nargs=2, type=float, default=[1.0, 3.0])
    ap.add_argument("--grid", type=json.loads, default=None,
                    help="JSON dict: parameter -> list of values (names must match the printed signature)")
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()

    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader
    cls = type(Plugins().get("pategan"))
    sig = inspect.signature(cls.__init__)
    print("synthcity PATE-GAN parameters:")
    for name, prm in sig.parameters.items():
        if name not in ("self", "kwargs"):
            print(f"  {name} = {prm.default!r}")

    grid = a.grid
    if grid is None:   # sensible defaults for the parameters that exist in this version
        cand = {"lamda": [None, 1e-2, 1e-1], "n_teachers": [None, 50]}
        grid = {k: v for k, v in cand.items() if k in sig.parameters}
        print(f"\nusing default grid {grid} (None = library default)")
    keys = list(grid)

    cfg, df, target, cat_cols = load(a.dataset, a.data_dir)
    sp = make_splits(df, target, seed=0)
    tr, va = sp["train"], sp["val"]
    feat = [c for c in df.columns if c != target]
    delta = min(1e-5, 1.0 / (10 * len(tr)))

    print(f"\n{'setting':40s} {'eps':>5s} {'fit s':>7s} {'AUPRC(val)':>10s}  output hash")
    for vals in itertools.product(*[grid[k] for k in keys]):
        kw = {k: v for k, v in zip(keys, vals) if v is not None}
        hashes = []
        for eps in a.eps:
            P.set_seed(0)
            t = time.time()
            try:
                gen = Plugins().get("pategan", epsilon=eps, delta=delta, random_state=0, **kw)
                gen.fit(GenericDataLoader(tr, target_column=target))
                syn = gen.generate(count=len(tr)).dataframe()[df.columns]
            except Exception as ex:
                print(f"{json.dumps(kw):40s} {eps:5g}  failed: {ex}")
                continue
            secs = time.time() - t
            h = hashlib.md5(syn.round(6).to_csv(index=False).encode()).hexdigest()[:10]
            hashes.append(h)
            y = (syn[target].astype(float) > 0.5).astype(int)
            score = float("nan")
            if y.nunique() == 2:
                rf = RandomForestClassifier(100, random_state=0, n_jobs=-1, class_weight="balanced")
                rf.fit(syn[feat].astype(float), y)
                score = average_precision_score(va[target], rf.predict_proba(va[feat].astype(float))[:, 1])
            print(f"{json.dumps(kw):40s} {eps:5g} {secs:7.1f} {score:10.3f}  {h}")
        if len(hashes) == 2:
            print(f"{'':40s} -> {'IDENTICAL outputs (budget has no effect)' if hashes[0] == hashes[1] else 'outputs differ (usable)'}")


if __name__ == "__main__":
    main()
