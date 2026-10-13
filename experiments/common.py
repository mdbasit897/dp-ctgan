"""Shared helpers for the experiment scripts: data splits and evaluation.

Every script (run_experiments.py, run_baselines.py, make_figures.py) uses
make_splits() so that all methods see exactly the same records for a seed.
"""
import os
import sys

import gpu_select  # must run before torch is imported: one GPU per process on multi-GPU machines

gpu_select.assign()

import numpy as np  # noqa: E402
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KernelDensity, KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import dp_ctgan_pipeline as P  # noqa: E402

DATASETS = ["adult", "taiwan_credit", "credit_fraud", "mimic3"]
DATASET_LABELS = {"adult": "Adult", "taiwan_credit": "Taiwan", "credit_fraud": "Credit Card Fraud",
                  "mimic3": "MIMIC-III"}


def eps_tag(eps):
    return "nodp" if eps is None else f"{eps:g}"


def parse_eps(s):
    return None if str(s).lower() in ("none", "nodp", "no-dp", "inf") else float(s)


# --------------------------------------------------------------------------- splits

def make_splits(df, target, seed, test_frac=0.20, holdout_frac=0.10, val_frac=0.10):
    """test (sealed, 20%) | pool 80% -> MIA holdout 10% of pool, validation 10% of pool,
    GAN training set = remaining 80% of pool (64% of all records). Stratified by label."""
    y = df[target].values.astype(int)
    idx = np.arange(len(df))
    i_pool, i_te = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=y)
    i_rest, i_ho = train_test_split(i_pool, test_size=holdout_frac, random_state=seed, stratify=y[i_pool])
    i_tr, i_val = train_test_split(i_rest, test_size=val_frac / (1 - holdout_frac),
                                   random_state=seed, stratify=y[i_rest])
    take = lambda ii: df.iloc[ii].reset_index(drop=True)
    return {"train": take(i_tr), "val": take(i_val), "holdout": take(i_ho), "test": take(i_te)}


def load(dataset, data_dir=None, verbose=False):
    cfg = P.DATASET_PRESETS[dataset]
    if data_dir:
        cfg = P.DatasetConfig(**{**cfg.__dict__, "path": os.path.join(data_dir, os.path.basename(cfg.path))})
    df, target, cat_cols = P.load_dataset(cfg, verbose=verbose)
    return cfg, df, target, cat_cols


# --------------------------------------------------------------------------- utility

def _scaled(df, scaler, feat, target):
    return scaler.transform(P._ensure_numeric(df, target)[feat].values.astype(float))


def best_threshold(y, p):
    """Threshold that maximises F1 on (y, p) — used on the VALIDATION split only."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.nanargmax(f1))]) if len(thr) else 0.5


def _clf_metrics(y_te, p_te, thr):
    pred = (p_te >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y_te, p_te)),
            "auprc": float(average_precision_score(y_te, p_te)),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred, zero_division=0)),
            "threshold": thr, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def tstr(train_df, splits, scaler, feat, target, knn_k=5, seed=0, return_probs=False):
    """Train RF and kNN on train_df (synthetic or real), choose the F1 threshold on the real
    validation split, report metrics on the sealed test split."""
    X_tr = _scaled(train_df, scaler, feat, target)
    y_tr = train_df[target].values.astype(int)
    X_val = _scaled(splits["val"], scaler, feat, target); y_val = splits["val"][target].values.astype(int)
    X_te = _scaled(splits["test"], scaler, feat, target); y_te = splits["test"][target].values.astype(int)
    out, probs = {}, {}
    if len(np.unique(y_tr)) < 2:
        nan = {k: float("nan") for k in ["auc", "auprc", "f1", "precision", "recall"]}
        return ({"rf": nan, "knn": nan}, {}) if return_probs else {"rf": nan, "knn": nan}
    models = {
        "rf": RandomForestClassifier(200, random_state=seed, n_jobs=-1, class_weight="balanced"),
        "knn": KNeighborsClassifier(n_neighbors=max(1, min(knn_k, len(X_tr) - 1)), n_jobs=-1),
    }
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        p_val, p_te = m.predict_proba(X_val)[:, 1], m.predict_proba(X_te)[:, 1]
        out[name] = _clf_metrics(y_te, p_te, best_threshold(y_val, p_val))
        probs[name] = p_te.astype(np.float32)
    return (out, probs) if return_probs else out


# --------------------------------------------------------------------------- fidelity

def fidelity(real_df, synth_df, feat, target, seed=0, n_disc=3000):
    ks, js = P.compute_ks_js(real_df, synth_df, feat)
    rm, sm = real_df[real_df[target] == 1], synth_df[synth_df[target] == 1]
    ks_min, js_min = P.compute_ks_js(rm, sm, feat) if len(sm) > 1 else (float("nan"), float("nan"))
    cr = real_df[feat].corr().values
    cs = synth_df[feat].corr().values
    iu = np.triu_indices(len(feat), 1)
    corr_diff = float(np.nanmean(np.abs(np.nan_to_num(cr) - np.nan_to_num(cs))[iu])) if len(feat) > 1 else 0.0
    # discriminator (propensity) AUC: 0.5 = indistinguishable
    rng = np.random.RandomState(seed)
    n = min(n_disc, len(real_df), len(synth_df))
    r = real_df[feat].iloc[rng.choice(len(real_df), n, replace=False)].values.astype(float)
    s = synth_df[feat].iloc[rng.choice(len(synth_df), n, replace=False)].values.astype(float)
    X = np.vstack([r, s]); y = np.r_[np.zeros(n), np.ones(n)]
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        rf = RandomForestClassifier(100, random_state=seed, n_jobs=-1).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], rf.predict_proba(X[te])[:, 1]))
    return {"ks": ks, "js": js, "ks_minority": ks_min, "js_minority": js_min,
            "corr_diff": corr_diff, "disc_auc": float(np.mean(aucs))}


# --------------------------------------------------------------------------- privacy

FPRS = (0.001, 0.01, 0.1)


def _attack_summary(y, s, y_cls=None):
    out = {"auc": float(roc_auc_score(y, s))}
    fpr, tpr, _ = roc_curve(y, s)
    for t in FPRS:
        ok = np.where(fpr <= t)[0]
        out[f"tpr@{t:g}"] = float(tpr[ok[-1]]) if len(ok) else 0.0
    if y_cls is not None:
        m = y_cls == 1
        if m.sum() > 4 and len(np.unique(y[m])) == 2:
            out["auc_minority"] = float(roc_auc_score(y[m], s[m]))
    return out


def privacy(splits, synth_df, feat, target, seed=0, n_eval=3000, return_scores=False):
    """Membership-inference evaluation on members (train) vs non-members (holdout).
    kNN-MIA (distance), DOMIAS-style density ratio (KDE, reference = validation split),
    LR-MIA (classifier confidence) and distance-to-closest-record (DCR) share."""
    rng = np.random.RandomState(seed)
    tr, ho = splits["train"], splits["holdout"]
    n = min(n_eval, len(tr), len(ho))
    mem = tr.iloc[rng.choice(len(tr), n, replace=False)]
    non = ho.iloc[rng.choice(len(ho), n, replace=False)]
    tgt = pd.concat([mem, non], ignore_index=True)
    y = np.r_[np.ones(n), np.zeros(n)]
    y_cls = tgt[target].values.astype(int)
    scores = {}

    sc = StandardScaler().fit(synth_df[feat].values.astype(float))
    Xs = sc.transform(synth_df[feat].values.astype(float))
    Xt = sc.transform(tgt[feat].values.astype(float))

    # kNN-MIA
    d, _ = NearestNeighbors(n_neighbors=min(5, len(Xs) - 1), n_jobs=-1).fit(Xs).kneighbors(Xt)
    scores["knn_mia"] = -d.mean(1)

    # DOMIAS-style: log p_synth(x) - log p_ref(x), densities by KDE in a PCA space fitted on reference data
    ref = splits["val"]
    sc_r = StandardScaler().fit(ref[feat].values.astype(float))
    pca = PCA(n_components=min(10, len(feat)), random_state=seed).fit(sc_r.transform(ref[feat].values.astype(float)))
    f = lambda D: pca.transform(sc_r.transform(D[feat].values.astype(float)))
    m_s = min(5000, len(synth_df))
    kde_s = KernelDensity(bandwidth="scott").fit(f(synth_df.iloc[rng.choice(len(synth_df), m_s, replace=False)]))
    kde_r = KernelDensity(bandwidth="scott").fit(f(ref))
    Zt = f(tgt)
    scores["domias"] = kde_s.score_samples(Zt) - kde_r.score_samples(Zt)

    out = {name: _attack_summary(y, s, y_cls) for name, s in scores.items()}
    out["lr_mia"] = P.lr_membership_inference_attack(tr, ho, synth_df, feat, seed=seed)

    # DCR share: fraction of synthetic records closer to a training record than to a holdout record
    sc_t = StandardScaler().fit(tr[feat].values.astype(float))
    m = min(2000, len(tr), len(ho))
    T = sc_t.transform(tr[feat].iloc[rng.choice(len(tr), m, replace=False)].values.astype(float))
    H = sc_t.transform(ho[feat].iloc[rng.choice(len(ho), m, replace=False)].values.astype(float))
    S = sc_t.transform(synth_df[feat].iloc[rng.choice(len(synth_df), min(5000, len(synth_df)), replace=False)]
                       .values.astype(float))
    d_t = NearestNeighbors(n_neighbors=1).fit(T).kneighbors(S)[0][:, 0]
    d_h = NearestNeighbors(n_neighbors=1).fit(H).kneighbors(S)[0][:, 0]
    out["dcr"] = {"share_closer_to_train": float(np.mean((d_t < d_h) + 0.5 * (d_t == d_h))),
                  "median_dcr_train": float(np.median(d_t)), "median_dcr_holdout": float(np.median(d_h))}
    if return_scores:
        return out, {"y": y.astype(np.int8), "y_cls": y_cls.astype(np.int8),
                     **{k: v.astype(np.float32) for k, v in scores.items()}}
    return out
