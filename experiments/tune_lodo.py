#!/usr/bin/env python3
"""Leave-one-dataset-out (LODO) hyperparameter selection for the end-to-end protocol.

The hyperparameters used for a dataset are chosen WITHOUT that dataset, so model selection spends none of
its privacy budget: the other three datasets play the role of public data.

  score   For every setting (No-DP, eps 1, 3, 10), every dataset and every candidate of one fixed candidate
          set (K configurations drawn once from the search space with a fixed seed), run RareSel-DP exactly
          as in the main grid (dp mode, private encoder) on the seed-0 split and record the validation RF
          AUPRC of the combined data (common + rare stream). The sealed test split and the MIA holdout are
          never used. Writes <out>/scores/<ds>_eps<e>_cand<k>.json; finished jobs are skipped, so the step
          can be restarted and sharded (--shard i/N).
  select  For a target dataset and setting, rank the candidates within each OTHER dataset (1 = best
          validation AUPRC), average the ranks over the other datasets and take the candidate with the best
          mean rank (ties: higher mean AUPRC). The target's own scores are not read. Writes
          <out>/<target>_eps<e>.json with best_params, which run_experiments.py reads via --hp-dir.

  python experiments/tune_lodo.py score --shard 0/8          # 8 workers, one per tmux window or with &
  python experiments/tune_lodo.py select
"""
import argparse
import glob
import json
import math
import os
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import DATASETS, P, eps_tag, load, make_splits, parse_eps

CANDIDATE_SEED = 2026
SETTINGS = ["none", "1", "3", "10"]


def candidates(k, seed=CANDIDATE_SEED):
    """K configurations drawn once from the search space. The same set is scored on every dataset and
    setting; dp_max_grad_norm applies under DP and r1_gamma without DP."""
    rng = np.random.RandomState(seed)
    logu = lambda lo, hi: float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    out = []
    for _ in range(k):
        out.append({"gan_epochs": int(rng.choice(np.arange(20, 101, 10))),
                    "gen_lr": logu(5e-5, 5e-4), "disc_lr": logu(5e-5, 5e-4),
                    "batch_size": int(rng.choice([256, 512])),
                    "dp_max_grad_norm": float(rng.uniform(1.0, 10.0)), "r1_gamma": float(rng.uniform(5.0, 15.0)),
                    "judge_keep_frac": float(rng.choice([0.25, 0.40])),
                    "judge_min_prob": float(rng.choice([0.01, 0.05]))})
    return out


def params_for(cand, eps):
    p = dict(cand)
    p.pop("r1_gamma" if eps is not None else "dp_max_grad_norm")
    return p


def score_one(ds, eps, k, cand, args):
    cfg, df, target, cat_cols = load(ds, args.data_dir)
    feat = [c for c in df.columns if c != target]
    sp = make_splits(df, target, seed=0)
    train, val = sp["train"], sp["val"]
    schema = P.public_schema(cfg, df.columns, target, cat_cols)
    hp = P.apply_privacy_mode(P.global_hyperparams(params_for(cand, eps)), "dp")
    hp.encoder = "dp"
    if args.quick:
        hp.gan_epochs, hp.flow_epochs, hp.finetune_epochs = 3, 3, 2
        hp.rare_pool_size, hp.rare_max_iters, hp.refine_steps, hp.flow_n_samples = 2000, 2, 5, 1000
    P.set_seed(0)
    t = time.time()
    model = P.DPCTGAN(epsilon=eps, hp=hp, verbose=False)
    model.fit(train, cat_cols, target, ds_name=ds, schema=schema)
    bounds, cat_sizes = model.release_bounds()
    post = lambda d: P.postprocess_synthetic(d, train, target, cat_cols, cfg.int_cols, cfg.hard_bounds,
                                             bounds=bounds, cat_sizes=cat_sizes)
    common = post(model.sample_common(len(train)))
    model.build_flows(hp)
    n_rare = max(200, int(model.stats_info["n_hat"] * hp.rare_multiplier))
    rare, diag = model.sample_rare(n_rare, hp=hp, return_diagnostics=True)
    rare[target] = 1
    combined = pd.concat([common, post(rare)], ignore_index=True)[df.columns]
    # validation-only evaluation (same classifier as the paper's TSTR; the test split is not touched)
    sc = StandardScaler().fit(P._ensure_numeric(train, target)[feat].values.astype(float))
    X = sc.transform(P._ensure_numeric(combined, target)[feat].values.astype(float))
    y = combined[target].values.astype(int)
    Xv = sc.transform(P._ensure_numeric(val, target)[feat].values.astype(float))
    yv = val[target].values.astype(int)
    if len(np.unique(y)) < 2:
        auprc = auc = float("nan")
    else:
        rf = RandomForestClassifier(200, random_state=0, n_jobs=-1, class_weight="balanced").fit(X, y)
        pv = rf.predict_proba(Xv)[:, 1]
        auprc, auc = float(average_precision_score(yv, pv)), float(roc_auc_score(yv, pv))
    return {"dataset": ds, "epsilon": eps, "candidate": k, "params": params_for(cand, eps),
            "val_auprc": auprc, "val_auc": auc, "achieved_epsilon": None if eps is None else model.epsilon_total,
            "n_selected": int(diag["n_selected"][0]) if len(diag.get("n_selected", [])) else None,
            "minutes": (time.time() - t) / 60, "quick": args.quick}


def cmd_score(args):
    cands = candidates(args.k)
    os.makedirs(os.path.join(args.out, "scores"), exist_ok=True)
    json.dump({"seed": CANDIDATE_SEED, "k": args.k, "candidates": cands},
              open(os.path.join(args.out, "candidates.json"), "w"), indent=1)
    jobs = [(ds, parse_eps(e), k) for ds in args.datasets for e in args.eps for k in range(args.k)]
    i, n = map(int, args.shard.split("/"))
    jobs = jobs[i::n]
    print(f"{len(jobs)} scoring jobs in shard {args.shard}")
    for ds, e, k in jobs:
        path = os.path.join(args.out, "scores", f"{ds}_eps{eps_tag(e)}_cand{k:02d}.json")
        if os.path.exists(path):
            continue
        print(f"=== {ds} eps={eps_tag(e)} candidate {k}", flush=True)
        try:
            R = score_one(ds, e, k, cands[k], args)
            json.dump(R, open(path + ".tmp", "w"), indent=1)
            os.replace(path + ".tmp", path)
            print(f"  val AUPRC {R['val_auprc']:.4f}  ({R['minutes']:.1f} min)", flush=True)
        except Exception:
            traceback.print_exc()
            open(path.replace(".json", ".error.txt"), "w").write(traceback.format_exc())
        finally:
            P._cuda_clear()


def cmd_select(args):
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(args.out, "scores", "*.json"))]
    if not rows:
        raise SystemExit(f"no scores in {args.out}/scores; run the score step first")
    S = pd.DataFrame(rows)
    S["eps"] = [eps_tag(None if pd.isna(x) else float(x)) for x in S["epsilon"]]
    cands = json.load(open(os.path.join(args.out, "candidates.json")))["candidates"]
    summary = []
    for e in args.eps:
        e_tag = eps_tag(parse_eps(e))
        s_e = S[S.eps == e_tag]
        for target in args.datasets:
            others = [d for d in args.datasets if d != target]
            missing = [(d, k) for d in others for k in range(len(cands))
                       if s_e[(s_e.dataset == d) & (s_e.candidate == k)].empty]
            if missing:
                print(f"skip {target} eps={e_tag}: {len(missing)} scores of the other datasets missing")
                continue
            ranks, means = [], []
            for d in others:
                s = s_e[s_e.dataset == d].set_index("candidate")["val_auprc"].reindex(range(len(cands)))
                ranks.append(s.fillna(-1).rank(ascending=False, method="average"))
                means.append(s)
            mean_rank = pd.concat(ranks, axis=1).mean(1)
            mean_auprc = pd.concat(means, axis=1).mean(1)
            order = sorted(range(len(cands)), key=lambda k: (mean_rank[k], -np.nan_to_num(mean_auprc[k], nan=-1)))
            best = order[0]
            out = {"dataset": target, "epsilon": parse_eps(e), "protocol": "lodo",
                   "selected_from": others, "candidate": int(best),
                   "best_params": params_for(cands[best], parse_eps(e)),
                   "mean_rank_other_datasets": float(mean_rank[best]),
                   "mean_val_auprc_other_datasets": float(mean_auprc[best]),
                   "n_candidates": len(cands)}
            path = os.path.join(args.out, f"{target}_eps{e_tag}.json")
            json.dump(out, open(path, "w"), indent=1)
            summary.append({"dataset": target, "eps": e_tag, "candidate": best, **out["best_params"],
                            "mean_rank": round(out["mean_rank_other_datasets"], 2)})
            print(f"  {path}: candidate {best}  mean rank {mean_rank[best]:.2f}  {out['best_params']}")
    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(args.out, "selected.csv"), index=False)
        S.sort_values(["dataset", "eps", "candidate"]).to_csv(os.path.join(args.out, "scores.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["score", "select"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--eps", nargs="+", default=SETTINGS)
    ap.add_argument("--k", type=int, default=24, help="number of candidate configurations")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="results_e2e/hparams_lodo")
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test (use a separate --out)")
    args = ap.parse_args()
    cmd_score(args) if args.step == "score" else cmd_select(args)


if __name__ == "__main__":
    main()
