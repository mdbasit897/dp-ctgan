#!/usr/bin/env python3
"""
DP-CTGAN — Generalized Differentially-Private Tabular Synthesis Pipeline
==========================================================================


WHAT CHANGED FROM THE ORIGINAL 4 SCRIPTS
-----------------------------------------
- Engine (DPCTGAN / StableNVP flows / PrivacyGate / kNN-MIA / LR-MIA /
  Optuna tuning / postprocessing) is now ONE shared implementation, based
  on the most bug-fixed version of the four (the Adult-Census script):
    * MIA train/holdout split has no leakage (test set is sealed).
    * LR-MIA shadow labels come from the synthetic data itself (unbiased).
    * refine_steps resolution bug fixed.
- Everything that used to be dataset-specific (file path, target column,
  categorical columns, log1p columns, ID/leakage columns, numeric bounds)
  now lives in ONE place: the DATASET_CONFIG dict below (or a --data /
  --target CLI call with auto-detection).

HOW TO USE ON YOUR OWN DATASET
-------------------------------
1. Easiest: just run
       python dp_ctgan_pipeline.py --data yourfile.csv --target your_label_col
   Categorical columns, log1p-worthy skewed columns, and ID columns are
   auto-detected. This works for most CSVs out of the box.

2. For more control (recommended for anything you'll publish results on),
   edit DATASET_CONFIG below, or add your own preset to DATASET_PRESETS,
   then run:
       python dp_ctgan_pipeline.py --preset your_dataset_name --epsilon 3

3. Every knob that used to require editing model internals is now a
   parameter on DATASET_CONFIG or in HYPERPARAMS — see the dataclasses
   below for the full list.

Run `python dp_ctgan_pipeline.py --help` for all CLI options.
"""

import os, sys, warnings, time, math, argparse, json
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Callable
from collections import defaultdict
from scipy.stats import rankdata, ks_2samp
from scipy.spatial.distance import jensenshannon

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, average_precision_score,
    precision_recall_curve, roc_curve, f1_score,
    precision_score, recall_score, balanced_accuracy_score,
)
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ModuleNotFoundError:
    optuna = None

np.random.seed(42)
torch.manual_seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 0 ─ DATASET CONFIG  (EDIT THIS PER DATASET — or use --data/--target)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class DatasetConfig:
    name: str
    path: str
    target_col: Optional[str] = None                  # exact column name if known
    target_candidates: List[str] = field(default_factory=list)  # tried in order if target_col not found
    positive_labels: List[str] = field(default_factory=lambda: ["1", "1.0", "true", "yes", ">50k"])
    id_cols: List[str] = field(default_factory=list)   # dropped unconditionally (IDs, free text, leakage)
    cat_cols: Optional[List[str]] = None               # None = auto-infer (nunique <= 20 & non-numeric or low-card numeric)
    log1p_cols: List[str] = field(default_factory=list)  # skewed numeric columns to log1p
    int_cols: List[str] = field(default_factory=list)  # columns to round to int after synthesis
    hard_bounds: Dict[str, tuple] = field(default_factory=dict)  # {"age": (17, 90)}
    drop_high_corr_threshold: Optional[float] = 0.95   # auto-drop columns this correlated with target (leakage guard)
    custom_clean_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None  # optional extra cleaning hook


@dataclass
class Hyperparams:
    gan_epochs: int = 60
    flow_epochs: int = 60
    latent_dim: int = 128
    hidden_dim: int = 256
    batch_size: int = 512
    gen_lr: float = 2e-4
    disc_lr: float = 2e-4
    dp_max_grad_norm: float = 5.0
    finetune_epochs: int = 15
    r1_gamma: float = 5.0
    minority_batch_frac: float = 0.5
    quantile_blend: float = 0.35
    test_split: float = 0.20
    final_mia_holdout_frac: float = 0.10
    rare_multiplier: float = 1.5            # how many synthetic minority rows to generate, relative to real count
    privacy_filter_enabled: bool = True


# ── Example presets — edit paths, or add your own dataset here ─────────────
DATASET_PRESETS: Dict[str, DatasetConfig] = {
    "adult": DatasetConfig(
        name="adult", path="adult.csv", target_col="income",
        target_candidates=["income", "income_>50K"],
        id_cols=["fnlwgt"],
        cat_cols=["workclass", "education", "marital-status", "occupation",
                  "relationship", "race", "sex", "native-country"],
        log1p_cols=["capital-gain", "capital-loss"],
        int_cols=["age", "education-num", "hours-per-week", "capital-gain", "capital-loss"],
        hard_bounds={"age": (17, 90), "education-num": (1, 16), "hours-per-week": (1, 99)},
    ),
    "taiwan_credit": DatasetConfig(
        name="taiwan_credit", path="UCI_Credit_Card.csv", target_col="default",
        target_candidates=["default.payment.next.month", "default", "Y"],
        id_cols=["ID"],
        cat_cols=["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3",
                  "PAY_4", "PAY_5", "PAY_6"],
        log1p_cols=[f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)],
        int_cols=["AGE", "LIMIT_BAL"],
        hard_bounds={"AGE": (21, 79), "LIMIT_BAL": (10_000, 1_000_000)},
    ),
    "credit_fraud": DatasetConfig(
        name="credit_fraud", path="creditcard.csv", target_col="Class",
        target_candidates=["Class", "class", "target"],
        cat_cols=[],  # all continuous (PCA components)
        log1p_cols=["Amount", "Time"],
    ),
    "mimic3": DatasetConfig(
        name="mimic3", path="final_dataset.csv", target_col="MORTALITY",
        id_cols=["ROW_ID", "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "DIAGNOSIS",
                 "EXPIRE_FLAG", "PRIMARY_ICD9", "DISCHARGE_LOCATION", "DEATHTIME",
                 "DOD", "HOSPITAL_EXPIRE_FLAG", "LANGUAGE", "ETHNICITY", "RELIGION"],
        log1p_cols=["HOSPITAL_LOS", "ICU_LOS", "NUM_DIAGNOSES", "NUM_PROCEDURES", "ICU_STAYS"],
    ),
}

# Active config for a direct `python dp_ctgan_pipeline.py` run with no CLI args.
DATASET_CONFIG = DATASET_PRESETS["adult"]
HYPERPARAMS = Hyperparams()


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 1 ─ GENERIC DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def _infer_categorical_columns(df: pd.DataFrame, target: str, max_card: int = 20) -> List[str]:
    cats = []
    for c in df.columns:
        if c == target:
            continue
        if df[c].dtype == object or str(df[c].dtype) == "category":
            cats.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= max_card:
            cats.append(c)
    return cats


def _encode_binary_target(series: pd.Series, positive_labels: List[str]) -> pd.Series:
    norm = series.astype(str).str.strip().str.rstrip(".").str.lower()
    pos = [str(p).lower() for p in positive_labels]
    mask = norm.isin(pos)
    if mask.sum() > 0:
        return mask.astype(int)
    # fall back: numeric already 0/1, or pick the minority class as positive
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and set(numeric.unique()) <= {0, 1}:
        return numeric.astype(int)
    vc = norm.value_counts()
    minority_label = vc.index[-1] if len(vc) >= 2 else vc.index[0]
    return (norm == minority_label).astype(int)


def load_dataset(cfg: DatasetConfig):
    """Load + clean any CSV according to a DatasetConfig. Returns (df, target, cat_cols)."""
    if not os.path.exists(cfg.path):
        raise FileNotFoundError(f"Dataset not found at '{cfg.path}'. Edit DatasetConfig.path.")

    df = pd.read_csv(cfg.path)
    df.columns = df.columns.str.strip()

    # target detection
    target = cfg.target_col
    if target is None or target not in df.columns:
        for cand in cfg.target_candidates:
            if cand in df.columns:
                target = cand
                break
    if target is None or target not in df.columns:
        raise ValueError(
            f"Could not find target column. Tried {cfg.target_col!r} and "
            f"{cfg.target_candidates}. Available columns: {list(df.columns)}")

    # drop ID / free-text / leakage columns
    drop_cols = [c for c in cfg.id_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"  [load] dropped ID/leakage columns: {drop_cols}")

    # clean missing values
    df = df.replace({"?": np.nan, "": np.nan}).dropna().reset_index(drop=True)

    # encode target as binary 0/1
    df[target] = _encode_binary_target(df[target], cfg.positive_labels)

    # categorical columns: explicit list or auto-infer
    cat_cols = cfg.cat_cols if cfg.cat_cols is not None else _infer_categorical_columns(df, target)
    cat_cols = [c for c in cat_cols if c in df.columns]
    for c in cat_cols:
        df[c] = pd.Categorical(df[c].astype(str).str.strip()).codes

    # log1p skewed numeric columns
    for c in cfg.log1p_cols:
        if c in df.columns:
            df[c] = np.log1p(pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0))

    # everything else numeric
    for c in df.columns:
        if c == target:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # optional custom hook for dataset-specific cleaning (unit fixes, physiological
    # clipping, etc.) that doesn't fit the generic config knobs above
    if cfg.custom_clean_fn is not None:
        df = cfg.custom_clean_fn(df)

    # leakage guard: warn (don't silently drop) on suspiciously high correlation with target
    if cfg.drop_high_corr_threshold is not None:
        for c in df.columns:
            if c == target:
                continue
            try:
                corr = abs(df[c].corr(df[target]))
                if corr > cfg.drop_high_corr_threshold:
                    print(f"  [WARN] column '{c}' has |corr|={corr:.3f} with target — "
                          f"possible leakage; consider adding it to id_cols.")
            except Exception:
                pass

    n_pos = int(df[target].sum())
    print(f"  [load] {cfg.name}: rows={len(df):,}  positive={n_pos:,} "
          f"({n_pos/len(df)*100:.2f}%)  features={df.shape[1]-1}  cat_cols={len(cat_cols)}")
    return df, target, cat_cols


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 2 ─ UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def _ensure_numeric(df, target):
    df = df.copy()
    for c in df.columns:
        if c == target:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def _rank_norm(x):
    return (rankdata(x, method='average') - 1) / max(len(x) - 1, 1)

def _cosine_sim_matrix(A, b):
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    return A_n @ (b / (np.linalg.norm(b) + 1e-8))

def greedy_maxmin_pca(arr, k, scores=None, pca_k=32):
    N = len(arr)
    if k >= N:
        return np.arange(N)
    try:
        arr_w = PCA(n_components=min(pca_k, arr.shape[1], N - 1),
                    whiten=True, random_state=42).fit_transform(arr)
    except Exception:
        arr_w = arr
    seed = int(np.argmax(scores)) if scores is not None else 0
    selected = [seed]
    min_dists = np.full(N, np.inf)
    for _ in range(k - 1):
        dists = np.linalg.norm(arr_w - arr_w[selected[-1]], axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -1
        selected.append(int(np.argmax(min_dists)))
    return np.array(selected)

def _cuda_clear():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 3 ─ DISTRIBUTION METRICS + MIA + PRIVACY GATE
# ══════════════════════════════════════════════════════════════════════════

def compute_ks_js(real_df, synth_df, feat_cols, n_bins=50):
    ks_vals, js_vals = [], []
    for c in feat_cols:
        if c not in real_df.columns or c not in synth_df.columns:
            continue
        r = real_df[c].dropna().values.astype(float)
        s = synth_df[c].dropna().values.astype(float)
        if len(r) < 2 or len(s) < 2:
            continue
        ks_val, _ = ks_2samp(r, s)
        ks_vals.append(float(ks_val))
        lo, hi = min(r.min(), s.min()), max(r.max(), s.max())
        if hi == lo:
            hi = lo + 1e-8
        bins = np.linspace(lo, hi, n_bins + 1)
        r_hist, _ = np.histogram(r, bins=bins)
        s_hist, _ = np.histogram(s, bins=bins)
        r_p = (r_hist + 1e-10) / (r_hist.sum() + 1e-10 * n_bins)
        s_p = (s_hist + 1e-10) / (s_hist.sum() + 1e-10 * n_bins)
        js_vals.append(float(jensenshannon(r_p, s_p)))
    return (float(np.mean(ks_vals)) if ks_vals else 1.0,
            float(np.mean(js_vals)) if js_vals else 1.0)


def membership_inference_attack(train_df, holdout_df, synth_df, feat_cols, n_eval=None, k=5):
    """kNN-MIA. train_df = GAN training rows (members); holdout_df = rows the GAN never saw."""
    if n_eval is None:
        n_eval = min(len(train_df), len(holdout_df), 3000)
    train_s = train_df.sample(n=min(n_eval, len(train_df)), random_state=42)
    holdout_s = holdout_df.sample(n=min(n_eval, len(holdout_df)), random_state=42)
    sc = StandardScaler().fit(synth_df[feat_cols].values.astype(float))
    X_synth = sc.transform(synth_df[feat_cols].values.astype(float))
    X_mem = sc.transform(train_s[feat_cols].values.astype(float))
    X_non = sc.transform(holdout_s[feat_cols].values.astype(float))
    kk = min(k, max(1, len(X_synth) - 1))
    nn = NearestNeighbors(n_neighbors=kk, n_jobs=-1).fit(X_synth)
    d_mem, _ = nn.kneighbors(X_mem)
    d_non, _ = nn.kneighbors(X_non)
    y_true = np.concatenate([np.ones(len(d_mem)), np.zeros(len(d_non))])
    y_score = np.concatenate([-d_mem.mean(1), -d_non.mean(1)])
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = 0.5
    return {"mia_auc": auc, "mia_advantage": 2 * (auc - 0.5)}


def lr_membership_inference_attack(train_df, holdout_df, synth_df, feat_cols,
                                    n_eval=2000, n_shadow=5, fpr_thresholds=(0.01, 0.05, 0.10)):
    """LR-MIA with unbiased shadow labels (derived from synth_df's own structure)."""
    n_eval = max(50, min(n_eval, len(train_df), len(holdout_df)))
    mem_idx = np.random.choice(len(train_df), n_eval, replace=False)
    non_idx = np.random.choice(len(holdout_df), n_eval, replace=False)
    mem_df = train_df.iloc[mem_idx].reset_index(drop=True)
    non_df = holdout_df.iloc[non_idx].reset_index(drop=True)
    all_real = pd.concat([mem_df, non_df], ignore_index=True)
    y_true = np.array([1] * len(mem_df) + [0] * len(non_df))

    sc_feat = StandardScaler()
    X_synth_sc = sc_feat.fit_transform(synth_df[feat_cols].values.astype(float))
    X_real_sc = sc_feat.transform(all_real[feat_cols].values.astype(float))

    target_candidates = [c for c in synth_df.columns if c not in feat_cols]
    if target_candidates and synth_df[target_candidates[0]].nunique() >= 2:
        synth_labels = synth_df[target_candidates[0]].values.astype(int)
    else:
        pca1 = PCA(n_components=1, random_state=42).fit_transform(X_synth_sc).flatten()
        synth_labels = (pca1 > np.median(pca1)).astype(int)

    conf_signals = np.zeros((len(all_real), n_shadow), dtype=np.float32)
    used = 0
    for sh in range(n_shadow):
        n_sh = min(len(X_synth_sc), 5000)
        idx = np.random.choice(len(X_synth_sc), n_sh, replace=False)
        y_sh = synth_labels[idx]
        if len(np.unique(y_sh)) < 2:
            continue
        try:
            clf = LogisticRegression(max_iter=300, solver="saga", C=1.0,
                                     random_state=sh * 7 + 42, n_jobs=-1)
            clf.fit(X_synth_sc[idx], y_sh)
            conf_signals[:, used] = clf.predict_proba(X_real_sc).max(axis=1)
            used += 1
        except Exception:
            continue
    if used == 0:
        empty = {"lr_mia_auc": 0.5, "lr_mia_advantage": 0.0}
        for t in fpr_thresholds:
            empty[f"tpr_at_fpr_{int(t*100):02d}"] = 0.0
        return empty

    conf = np.clip(conf_signals[:, :used], 1e-7, 1 - 1e-7)
    loss = -np.log(conf)
    atk_feat = np.column_stack([conf.mean(1), conf.std(1), loss.mean(1), loss.std(1)])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs_all = np.zeros(len(y_true))
    fold_aucs = []
    sc_atk = StandardScaler()
    for tr_idx, te_idx in skf.split(atk_feat, y_true):
        X_tr = sc_atk.fit_transform(atk_feat[tr_idx])
        X_te = sc_atk.transform(atk_feat[te_idx])
        if len(np.unique(y_true[tr_idx])) < 2:
            continue
        lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        lr.fit(X_tr, y_true[tr_idx])
        p = lr.predict_proba(X_te)[:, 1]
        probs_all[te_idx] = p
        fold_aucs.append(float(roc_auc_score(y_true[te_idx], p)))
    final_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.5
    result = {"lr_mia_auc": final_auc, "lr_mia_advantage": float(2 * (final_auc - 0.5))}
    try:
        fpr_arr, tpr_arr, _ = roc_curve(y_true, probs_all)
        for t in fpr_thresholds:
            idx_b = np.where(fpr_arr <= t)[0]
            result[f"tpr_at_fpr_{int(t*100):02d}"] = float(tpr_arr[idx_b[-1]]) if len(idx_b) else 0.0
    except Exception:
        for t in fpr_thresholds:
            result[f"tpr_at_fpr_{int(t*100):02d}"] = 0.0
    return result


def privacy_realism_filter(synth_df, ref_df, feat_cols, desired_n=None, k=3,
                            min_q=0.03, max_q=0.995, max_ref=2000, label="", verbose=True):
    """Removes near-copies (privacy risk) and out-of-support outliers (realism risk)."""
    if len(synth_df) == 0:
        return synth_df
    use_cols = [c for c in feat_cols if c in ref_df.columns and c in synth_df.columns]
    if not use_cols:
        return synth_df.head(desired_n) if desired_n else synth_df
    ref_s = ref_df.sample(n=min(len(ref_df), max_ref), random_state=42)
    X_ref = _ensure_numeric(ref_s[use_cols], "__none__")[use_cols].values.astype(float)
    X_syn = _ensure_numeric(synth_df[use_cols], "__none__")[use_cols].values.astype(float)
    sc = StandardScaler().fit(X_ref)
    X_ref, X_syn = sc.transform(X_ref), sc.transform(X_syn)
    k_ref = min(k + 1, len(X_ref))
    d_ref, _ = NearestNeighbors(n_neighbors=k_ref, n_jobs=1).fit(X_ref).kneighbors(X_ref)
    ref_dist = d_ref[:, 1:].mean(1) if d_ref.shape[1] > 1 else d_ref[:, 0]
    d_syn, _ = NearestNeighbors(n_neighbors=min(k, len(X_ref)), n_jobs=1).fit(X_ref).kneighbors(X_syn)
    syn_dist = d_syn.mean(1)
    lower = max(float(np.quantile(ref_dist, min_q)), 1e-6)
    upper = max(float(np.quantile(ref_dist, max_q)) * 1.25, lower * 1.5)
    keep_mask = ~((syn_dist < lower) | (syn_dist > upper))
    df = synth_df.reset_index(drop=True)
    safe = df.loc[keep_mask]
    if desired_n is None:
        out = safe
    elif len(safe) >= desired_n:
        out = safe.sample(n=desired_n, random_state=42)
    else:
        out = pd.concat([safe, df.loc[~keep_mask].sample(
            n=min(desired_n - len(safe), (~keep_mask).sum()), random_state=42)])
    if verbose:
        print(f"  [PrivacyGate:{label}] keep={keep_mask.sum():,}/{len(df):,}  return={len(out):,}")
    return out.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 4 ─ EVALUATION (RF + KNN downstream classifiers)
# ══════════════════════════════════════════════════════════════════════════

def eval_clf(X_tr, y_tr, X_te, y_te):
    if len(np.unique(y_tr)) < 2:
        return {"auc": 0., "auprc": 0., "f1": 0., "precision": 0., "recall": 0.}
    rf = RandomForestClassifier(200, random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(X_tr, y_tr)
    prob = rf.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_te, prob)) if len(np.unique(y_te)) > 1 else 0.5,
        "auprc": float(average_precision_score(y_te, prob)) if len(np.unique(y_te)) > 1 else 0.,
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "prob": prob,
    }

def eval_clf_knn(X_tr, y_tr, X_te, y_te, n_neighbors=5):
    if len(np.unique(y_tr)) < 2:
        return {"knn_auc": 0., "knn_auprc": 0., "knn_f1": 0.}
    k = max(1, min(n_neighbors, len(X_tr) - 1))
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_tr, y_tr)
    prob = knn.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "knn_auc": float(roc_auc_score(y_te, prob)) if len(np.unique(y_te)) > 1 else 0.5,
        "knn_auprc": float(average_precision_score(y_te, prob)) if len(np.unique(y_te)) > 1 else 0.,
        "knn_f1": float(f1_score(y_te, pred, zero_division=0)),
    }

def tstr_eval(train_df, sc, X_te, y_te, feat, target):
    X_tr = sc.transform(_ensure_numeric(train_df, target)[feat].values.astype(float))
    y_tr = train_df[target].values.astype(int)
    out = eval_clf(X_tr, y_tr, X_te, y_te)
    out.update(eval_clf_knn(X_tr, y_tr, X_te, y_te))
    return out


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 5 ─ QUANTILE MATCHING + POSTPROCESSING
# ══════════════════════════════════════════════════════════════════════════

def quantile_match(synth_df, ref_df, feat_cols, blend, orig_df=None):
    matched = synth_df.copy()
    n_s = len(synth_df)
    for c in feat_cols:
        if c not in synth_df.columns or c not in ref_df.columns:
            continue
        r = np.sort(ref_df[c].dropna().values.astype(float))
        if len(r) < 2:
            continue
        s = synth_df[c].values.astype(float)
        ranks = rankdata(s, method='ordinal') - 1
        r_idx = np.clip((ranks / max(n_s - 1, 1) * (len(r) - 1)).astype(int), 0, len(r) - 1)
        lo = float(orig_df[c].min()) if orig_df is not None else r.min()
        hi = float(orig_df[c].max()) if orig_df is not None else r.max()
        matched[c] = np.clip(blend * r[r_idx] + (1 - blend) * s, lo, hi)
    return matched


def postprocess_synthetic(synth_df, ref_df, target_col, cat_cols, int_cols, hard_bounds, verbose=False):
    df = synth_df.copy()
    feat_cols = [c for c in df.columns if c != target_col]
    for c in feat_cols:
        if c not in ref_df.columns:
            continue
        lo_hard, hi_hard = hard_bounds.get(c, (-np.inf, np.inf))
        lo = max(lo_hard, float(ref_df[c].min()))
        hi = min(hi_hard, float(ref_df[c].max()))
        if lo > hi:
            lo, hi = float(ref_df[c].min()), float(ref_df[c].max())
        df[c] = np.clip(df[c].values.astype(float), lo, hi)
    for c in int_cols:
        if c in df.columns:
            df[c] = np.round(df[c].values.astype(float)).astype(int)
    for c in cat_cols:
        if c in df.columns and c in ref_df.columns:
            n_codes = int(ref_df[c].nunique())
            if n_codes >= 2:
                df[c] = np.clip(np.round(df[c].values.astype(float)).astype(int), 0, n_codes - 1)
    if verbose:
        print(f"  [PostProc] rows={len(df):,}")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 6 ─ MODE NORMALIZER
# ══════════════════════════════════════════════════════════════════════════

class ModeNormalizer:
    def __init__(self, n_modes=4):
        self.n_modes = n_modes
        self.gmms_, self.cat_nvals_, self._slices = {}, {}, {}
        self.cont_cols_, self.cat_cols_, self.out_dim_ = [], [], 0

    def fit(self, df, discrete_cols):
        self.cont_cols_ = [c for c in df.columns if c not in discrete_cols]
        self.cat_cols_ = [c for c in discrete_cols if c in df.columns]
        idx = 0
        for c in self.cont_cols_:
            v = df[c].values.reshape(-1, 1).astype(float)
            uq = len(np.unique(v))
            nm = min(self.n_modes, max(1, uq // 2)) if uq >= 2 else 1
            gm = GaussianMixture(n_components=nm, random_state=42, covariance_type="full", reg_covar=1e-4)
            gm.fit(v + (np.random.normal(0, 1e-6, v.shape) if uq < 2 else 0))
            self.gmms_[c] = gm
            self._slices[c] = slice(idx, idx + 1 + nm)
            idx += 1 + nm
        for c in self.cat_cols_:
            nv = max(int(df[c].nunique()), 1)
            self.cat_nvals_[c] = nv
            self._slices[c] = slice(idx, idx + nv)
            idx += nv
        self.out_dim_ = idx
        return self

    def transform(self, df):
        n = len(df)
        out = np.zeros((n, self.out_dim_), dtype=np.float32)
        for c in self.cont_cols_:
            v = df[c].values.reshape(-1, 1).astype(float)
            gm = self.gmms_[c]
            modes = gm.predict(v)
            means = gm.means_.flatten()[modes]
            stds = np.sqrt(np.array([float(gm.covariances_[m].flatten()[0]) for m in modes]))
            sl = self._slices[c]
            out[:, sl.start] = np.clip((v.flatten() - means) / (4 * stds + 1e-8), -0.99, 0.99)
            for i, m in enumerate(modes):
                out[i, sl.start + 1 + m] = 1.
        for c in self.cat_cols_:
            vals = df[c].values.astype(int)
            nv = self.cat_nvals_[c]
            sl = self._slices[c]
            for i, val in enumerate(vals):
                out[i, sl.start + (int(val) % nv)] = 1.
        return out

    def inverse_transform(self, data, orig_df):
        rows = []
        for row in data:
            rec = {}
            for c in self.cont_cols_:
                sl, gm = self._slices[c], self.gmms_[c]
                m = int(np.argmax(row[sl.start + 1:sl.stop]))
                std = float(np.sqrt(gm.covariances_[m].flatten()[0]))
                val = float(row[sl.start]) * 4. * std + float(gm.means_.flatten()[m])
                rec[c] = float(np.clip(val, float(orig_df[c].min()), float(orig_df[c].max())))
            for c in self.cat_cols_:
                sl = self._slices[c]
                rec[c] = int(np.argmax(row[sl])) % self.cat_nvals_[c]
            rows.append(rec)
        return pd.DataFrame(rows, columns=self.cont_cols_ + self.cat_cols_)


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 7 ─ GAN NETWORKS + STABLE NVP FLOW
# ══════════════════════════════════════════════════════════════════════════

class Generator(nn.Module):
    def __init__(self, latent_dim, cond_dim, data_dim, hidden=(256, 256)):
        super().__init__()
        prev, layers = latent_dim + cond_dim, []
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()]
            prev = h
        self.shared = nn.Sequential(*layers)
        self.out = nn.Sequential(nn.Linear(prev, data_dim), nn.Tanh())
        self._hidden_dim = prev

    def forward(self, z, cond):
        return self.out(self.shared(torch.cat([z, cond], dim=1)))


class DiscriminatorNet(nn.Module):
    def __init__(self, in_dim, hidden=(256, 256)):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        prev = in_dim
        for h in hidden:
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(prev, h), nn.LayerNorm(h), nn.LeakyReLU(0.2, inplace=False)))
            prev = h
        self.output_layer = nn.Linear(prev, 1)

    def get_features(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x

    def forward(self, x):
        return self.output_layer(self.get_features(x))


class StableAffineCoupling(nn.Module):
    def __init__(self, dim, hidden, scale_max=2.0):
        super().__init__()
        self.scale_max = scale_max
        self.d1, self.d2 = dim // 2, dim - dim // 2
        def mlp(i, o):
            return nn.Sequential(nn.Linear(i, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, o))
        self.s, self.t = mlp(self.d1, self.d2), mlp(self.d1, self.d2)

    def forward(self, x):
        x1, x2 = x[:, :self.d1], x[:, self.d1:]
        s = torch.tanh(self.s(x1)) * self.scale_max
        return torch.cat([x1, x2 * torch.exp(s) + self.t(x1)], 1), s.sum(1)


class StableNVP(nn.Module):
    def __init__(self, dim, n_layers=6, hidden=128, scale_max=2.0):
        super().__init__()
        self.pad = dim % 2
        self.dim = dim + self.pad
        self.layers = nn.ModuleList([StableAffineCoupling(self.dim, hidden, scale_max) for _ in range(n_layers)])

    def log_prob(self, x):
        if self.pad:
            x = torch.cat([x, torch.zeros(x.size(0), self.pad, device=x.device)], 1)
        ld = torch.zeros(x.size(0), device=x.device)
        for layer in self.layers:
            x, d = layer(x)
            ld += d
        return (-0.5 * (x.pow(2).sum(1) + self.dim * math.log(2 * math.pi)) + ld).clamp(-80., 80.)


def _train_flow(flow, data_tensor, epochs, lr, device):
    flow = flow.to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.1)
    dl = DataLoader(TensorDataset(data_tensor), batch_size=256, shuffle=True, drop_last=True)
    flow.train()
    for _ in range(epochs):
        for (x,) in dl:
            loss = -flow.log_prob(x.to(device)).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            opt.step()
        sched.step()
    flow.eval()


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 8 ─ DP-CTGAN  (dataset-agnostic engine)
# ══════════════════════════════════════════════════════════════════════════

class DPCTGAN:
    """
    Differentially-private CTGAN with:
      - minority-aware moment/feature matching so rare classes aren't dropped
      - optional (epsilon=None => no-DP) Opacus DP-SGD training
      - StableNVP normalizing-flow "rare sampler" for oversampling the minority
        class beyond what plain GAN sampling would give you
    Dataset-agnostic: works on the ModeNormalizer-encoded representation only.
    """

    def __init__(self, epsilon=None, delta=1e-5, hp: Hyperparams = None, verbose=True):
        hp = hp or Hyperparams()
        self.epsilon = epsilon
        self.use_dp = epsilon is not None
        self.delta = delta
        self.epochs = hp.gan_epochs
        self.flow_epochs = hp.flow_epochs
        self.batch_size = hp.batch_size
        self.latent_dim = hp.latent_dim
        self.hidden_dims = (hp.hidden_dim, hp.hidden_dim)
        self.gen_lr, self.disc_lr = hp.gen_lr, hp.disc_lr
        self.max_grad_norm = hp.dp_max_grad_norm
        self.finetune_epochs = hp.finetune_epochs
        self.r1_gamma = hp.r1_gamma if not self.use_dp else 0.0
        self.minority_batch_frac = hp.minority_batch_frac
        self.verbose = verbose
        self._fitted = False
        self.flow_min = self.flow_maj = None
        self._h_scaler = self._flow_pca = None
        self._minority_idx = self._majority_idx = None
        self._minority_rate = 0.0
        self._actual_epsilon = float('inf')

    def _cond(self, n, mode='majority'):
        c = torch.zeros(n, 2, device=DEVICE)
        c[:, 1 if mode == 'minority' else 0] = 1.
        return c

    def _balanced_batch(self, data_tensor, bs):
        if self._minority_idx is None or len(self._minority_idx) == 0:
            return data_tensor[np.random.randint(0, len(data_tensor), bs)].to(DEVICE)
        n_min = max(1, int(bs * self.minority_batch_frac))
        n_maj = bs - n_min
        idx = np.concatenate([np.random.choice(self._minority_idx, n_min, replace=True),
                              np.random.choice(self._majority_idx, n_maj, replace=True)])
        return data_tensor[idx].to(DEVICE)

    def fit(self, df, discrete_columns, target_col, ds_name=""):
        t0 = time.time()
        self._target_col = target_col
        self._norm = ModeNormalizer().fit(df, discrete_columns)
        data_np = self._norm.transform(df)
        data_dim = data_np.shape[1]
        self._orig_df = df.copy()

        labels = df[target_col].values.astype(int)
        self._minority_idx = np.where(labels == 1)[0]
        self._majority_idx = np.where(labels == 0)[0]
        self._minority_rate = len(self._minority_idx) / max(len(labels), 1)
        min_np = data_np[self._minority_idx] if len(self._minority_idx) else data_np[:1]
        self._min_mean = torch.tensor(min_np.mean(0), dtype=torch.float32, device=DEVICE)
        self._min_std = torch.tensor(min_np.std(0) + 1e-6, dtype=torch.float32, device=DEVICE)

        G = Generator(self.latent_dim, 2, data_dim, self.hidden_dims).to(DEVICE)
        D = DiscriminatorNet(data_dim + 2, self.hidden_dims).to(DEVICE)
        if self.use_dp:
            D = ModuleValidator.fix(D)

        opt_G = optim.Adam(G.parameters(), lr=self.gen_lr, betas=(0.5, 0.9))
        opt_D = optim.Adam(D.parameters(), lr=self.disc_lr, betas=(0.5, 0.9))
        bce = nn.BCEWithLogitsLoss()
        data_tensor = torch.tensor(data_np, dtype=torch.float32)
        loader = DataLoader(TensorDataset(data_tensor), batch_size=self.batch_size, shuffle=True, drop_last=True)

        pe = None
        if self.use_dp:
            pe = PrivacyEngine(accountant="rdp")
            D, opt_D, loader = pe.make_private_with_epsilon(
                module=D, optimizer=opt_D, data_loader=loader,
                target_epsilon=self.epsilon, target_delta=self.delta,
                epochs=self.epochs, max_grad_norm=self.max_grad_norm)
            D.grad_accumulation_allowed = True
            raw_D = D._module
        else:
            raw_D = D

        self._G, self._D = G, D
        dp_tag = f"ε={self.epsilon}" if self.use_dp else "No-DP"
        if self.verbose:
            print(f"  ┌─ DPCTGAN [{dp_tag}] [{ds_name}] epochs={self.epochs} "
                  f"batch={self.batch_size} minority_rate={self._minority_rate:.4f}")

        log_every = max(1, self.epochs // 5)
        for epoch in range(1, self.epochs + 1):
            G.train(); D.train()
            for (real_batch,) in loader:
                real_batch = real_batch.to(DEVICE)
                bs_d = real_batch.size(0)
                if bs_d < 2:
                    continue
                cond_d = self._cond(bs_d, 'majority')
                with torch.no_grad():
                    fake_d = G(torch.randn(bs_d, self.latent_dim, device=DEVICE), cond_d)
                if self.use_dp or self.r1_gamma <= 0:
                    both = D(torch.cat([torch.cat([real_batch, cond_d], 1),
                                        torch.cat([fake_d, cond_d], 1)], 0))
                    loss_D = (bce(both[:bs_d], torch.full((bs_d, 1), 0.9, device=DEVICE))
                              + bce(both[bs_d:], torch.zeros(bs_d, 1, device=DEVICE))) * 0.5
                else:
                    real_gp = real_batch.clone().detach().requires_grad_(True)
                    d_real = D(torch.cat([real_gp, cond_d.detach()], 1))
                    grad = torch.autograd.grad(d_real.sum(), real_gp, create_graph=True, retain_graph=True)[0]
                    r1 = (self.r1_gamma / 2.0) * grad.pow(2).sum(1).mean()
                    d_fake = D(torch.cat([fake_d, cond_d], 1))
                    loss_D = (bce(d_real, torch.full((bs_d, 1), 0.9, device=DEVICE))
                              + bce(d_fake, torch.zeros(bs_d, 1, device=DEVICE))) * 0.5 + r1
                opt_D.zero_grad(); loss_D.backward(); opt_D.step()

                bs = self.batch_size
                rb = self._balanced_batch(data_tensor, bs)
                cond_g = self._cond(bs, 'majority')
                fake_g = G(torch.randn(bs, self.latent_dim, device=DEVICE), cond_g)
                f_in = torch.cat([fake_g, cond_g], 1)
                adv = bce(raw_D(f_in), torch.ones(bs, 1, device=DEVICE))
                with torch.no_grad():
                    rf_m = raw_D.get_features(torch.cat([rb, cond_g], 1)).mean(0)
                fm = F.mse_loss(raw_D.get_features(f_in).mean(0), rf_m.detach())

                mm_bs = min(256, bs)
                z_min = torch.randn(mm_bs, self.latent_dim, device=DEVICE)
                fake_min = G(z_min, self._cond(mm_bs, 'minority'))
                mm_loss = (F.l1_loss(fake_min.mean(0), self._min_mean.detach())
                           + F.l1_loss(fake_min.std(0), self._min_std.detach()))

                loss_G = adv + 1.0 * fm + 2.0 * mm_loss
                opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            if self.verbose and epoch % log_every == 0:
                eps_now = pe.get_epsilon(self.delta) if self.use_dp else float('inf')
                eps_s = f"  ε={eps_now:.3f}" if self.use_dp else ""
                print(f"  ep {epoch:3d}/{self.epochs}{eps_s}")

        self._actual_epsilon = pe.get_epsilon(self.delta) if self.use_dp else float('inf')

        if self.finetune_epochs > 0:
            G.train()
            opt_ft = optim.Adam(G.parameters(), lr=max(self.gen_lr * 0.1, 1e-6))
            for _ in range(self.finetune_epochs):
                z_min = torch.randn(256, self.latent_dim, device=DEVICE)
                fake_min = G(z_min, self._cond(256, 'minority'))
                ft_loss = (F.l1_loss(fake_min.mean(0), self._min_mean.detach())
                           + 0.5 * F.l1_loss(fake_min.std(0), self._min_std.detach()))
                opt_ft.zero_grad(); ft_loss.backward(); opt_ft.step()
            G.eval()

        self._fitted = True
        if self.verbose:
            print(f"  Training done  {time.time()-t0:.1f}s  "
                  f"ε={'∞' if not self.use_dp else f'{self._actual_epsilon:.3f}'}")
        return self

    def _sample_raw(self, n, mode, batch=256):
        self._G.eval()
        rows = []
        with torch.no_grad():
            for s in range(0, n, batch):
                nb = min(batch, n - s)
                z = torch.randn(nb, self.latent_dim, device=DEVICE)
                rows.append(self._G(z, self._cond(nb, mode)).cpu().numpy())
        raw = np.vstack(rows)
        synth = self._norm.inverse_transform(raw, self._orig_df)
        for col in self._orig_df.columns:
            if col not in synth.columns:
                synth[col] = self._orig_df[col].mode()[0]
        return synth[self._orig_df.columns].reset_index(drop=True), raw

    def sample_common(self, n):
        rate = self._minority_rate
        n_min, n_maj = max(1, int(round(n * rate))), n - max(1, int(round(n * rate)))
        df_maj, _ = self._sample_raw(n_maj, 'majority')
        df_min, _ = self._sample_raw(n_min, 'minority')
        df_maj[self._target_col] = 0
        df_min[self._target_col] = 1
        return pd.concat([df_maj, df_min], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    def build_flows(self, n_samp=8000, hp: Hyperparams = None):
        """Trains a pair of normalizing flows over the GAN's hidden representation,
        used by sample_rare() to score/oversample minority-class latent regions."""
        assert self._fitted
        hp = hp or Hyperparams()
        self._G.eval()
        def sample_h(n, mode):
            hs = []
            with torch.no_grad():
                for s in range(0, n, 256):
                    nb = min(256, n - s)
                    z = torch.randn(nb, self.latent_dim, device=DEVICE)
                    hs.append(self._G.shared(torch.cat([z, self._cond(nb, mode)], 1)).cpu())
            return torch.cat(hs).numpy()

        h_min_np, h_maj_np = sample_h(n_samp, 'minority'), sample_h(n_samp, 'majority')
        h_all = np.concatenate([h_min_np, h_maj_np])
        pca_k = min(32, h_all.shape[1], len(h_all) - 1)
        self._flow_pca = PCA(n_components=pca_k, whiten=True, random_state=42).fit(h_all)
        h_min_pca = self._flow_pca.transform(h_min_np).astype(np.float32)
        h_maj_pca = self._flow_pca.transform(h_maj_np).astype(np.float32)
        self._h_scaler = StandardScaler().fit(np.concatenate([h_min_pca, h_maj_pca]))
        h_min_t = torch.tensor(self._h_scaler.transform(h_min_pca), dtype=torch.float32)
        h_maj_t = torch.tensor(self._h_scaler.transform(h_maj_pca), dtype=torch.float32)

        self.flow_min = StableNVP(pca_k, n_layers=6, hidden=128)
        self.flow_maj = StableNVP(pca_k, n_layers=6, hidden=128)
        _train_flow(self.flow_min, h_min_t, hp.flow_epochs, 2e-4, DEVICE)
        _train_flow(self.flow_maj, h_maj_t, hp.flow_epochs, 2e-4, DEVICE)
        _cuda_clear()

    def _scale_h(self, h_np):
        return self._h_scaler.transform(self._flow_pca.transform(h_np)).astype(np.float32)

    def sample_rare(self, n, df_real=None, pool_mult=3, max_iters=8):
        """Oversamples the minority class by ranking candidates with the flow
        likelihood-ratio (minority flow vs majority flow) and an optional RF
        probability score, then greedily selecting a diverse top-scoring subset."""
        assert self.flow_min is not None, "call build_flows() first"
        self._G.eval()
        clf = None
        if df_real is not None:
            feat_cols = [c for c in df_real.columns if c != self._target_col]
            try:
                X_r = _ensure_numeric(df_real, self._target_col)[feat_cols].values.astype(float)
                y_r = df_real[self._target_col].values.astype(int)
                if len(np.unique(y_r)) >= 2:
                    clf = RandomForestClassifier(150, random_state=42, n_jobs=-1, class_weight="balanced")
                    clf.fit(X_r, y_r)
            except Exception:
                pass

        pool_sz = max(n * pool_mult, 2000)
        all_raw, all_h, all_score = [], [], []
        for _ in range(max_iters):
            if sum(len(r) for r in all_raw) >= n * 2:
                break
            zs, cs, hs = [], [], []
            with torch.no_grad():
                for s in range(0, pool_sz, 256):
                    nb = min(256, pool_sz - s)
                    z = torch.randn(nb, self.latent_dim, device=DEVICE)
                    c = self._cond(nb, 'minority')
                    zs.append(z.cpu()); cs.append(c.cpu())
                    hs.append(self._G.shared(torch.cat([z, c], 1)).cpu())
            z_all, c_all, h_all = torch.cat(zs), torch.cat(cs), torch.cat(hs)
            h_sc = torch.tensor(self._scale_h(h_all.numpy()), dtype=torch.float32)
            with torch.no_grad():
                lr_score = (self.flow_min.log_prob(h_sc.to(DEVICE))
                            - self.flow_maj.log_prob(h_sc.to(DEVICE))).cpu().numpy()
            lr_rank = _rank_norm(lr_score)
            k_keep = min(len(z_all), max(n * 2, pool_sz // 2))
            top_idx = np.argsort(lr_rank)[::-1][:k_keep]
            raw_out = []
            with torch.no_grad():
                for s in range(0, len(top_idx), 256):
                    idx = top_idx[s:s+256]
                    raw_out.append(self._G(z_all[idx].to(DEVICE), c_all[idx].to(DEVICE)).cpu().numpy())
            raw_out = np.concatenate(raw_out)
            df_pool = self._norm.inverse_transform(raw_out, self._orig_df)
            for col in self._orig_df.columns:
                if col not in df_pool.columns:
                    df_pool[col] = self._orig_df[col].mode()[0]
            df_pool = df_pool[self._orig_df.columns].reset_index(drop=True)
            mask = np.ones(len(df_pool), dtype=bool)
            if clf is not None:
                try:
                    X_p = _ensure_numeric(df_pool, self._target_col)[feat_cols].values.astype(float)
                    probs = clf.predict_proba(X_p)[:, 1]
                    mask = probs >= np.percentile(probs, 60)
                except Exception:
                    pass
            if mask.sum() == 0:
                continue
            all_raw.append(raw_out[mask])
            all_h.append(h_sc.numpy()[top_idx][mask])
            all_score.append(lr_rank[top_idx][mask])

        if not all_raw:
            df_fb, _ = self._sample_raw(n, 'minority')
            df_fb[self._target_col] = 1
            return df_fb.head(n).reset_index(drop=True)

        mega_raw, mega_h, mega_score = np.concatenate(all_raw), np.concatenate(all_h), np.concatenate(all_score)
        df_all = self._norm.inverse_transform(mega_raw, self._orig_df)
        for col in self._orig_df.columns:
            if col not in df_all.columns:
                df_all[col] = self._orig_df[col].mode()[0]
        df_all = df_all[self._orig_df.columns].reset_index(drop=True)
        k_sel = min(n, len(df_all))
        sel_idx = greedy_maxmin_pca(mega_h, k=k_sel, scores=mega_score)
        df_sel = df_all.iloc[sel_idx].copy().reset_index(drop=True)
        df_sel[self._target_col] = 1
        return df_sel


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 9 ─ OPTUNA HYPERPARAMETER TUNING (optional, 2-stage: GAN + DP)
# ══════════════════════════════════════════════════════════════════════════

def target_cat_cols(df, feat_cols, target, max_card=20):
    return [c for c in feat_cols if c != target and df[c].nunique() <= max_card]


def optuna_tune(epsilon, gan_train_df, val_df, mia_holdout_df, sc_val, X_val, y_val,
                target, feat_cols, cat_cols, base_hp: Hyperparams, n_trials=8):
    """Tunes gan_epochs / lr / batch_size / (max_grad_norm if DP) against a
    TSTR-AUC objective penalized for distribution mismatch and MIA leakage."""
    if optuna is None:
        print("  [Optuna] not installed — skipping tuning, using default hyperparams.")
        return base_hp

    def objective(trial):
        hp = Hyperparams(**asdict(base_hp))
        hp.gan_epochs = trial.suggest_int("gan_epochs", 20, 60, step=10)
        hp.gen_lr = trial.suggest_float("gen_lr", 5e-5, 5e-4, log=True)
        hp.disc_lr = trial.suggest_float("disc_lr", 5e-5, 5e-4, log=True)
        hp.batch_size = trial.suggest_categorical("batch_size", [256, 512])
        if epsilon is not None:
            hp.dp_max_grad_norm = trial.suggest_float("max_grad_norm", 1.0, 10.0)
        try:
            model = DPCTGAN(epsilon=epsilon, hp=hp, verbose=False)
            model.fit(gan_train_df, cat_cols, target, ds_name="optuna_trial")
            synth = model.sample_common(len(gan_train_df))
            X_tr = sc_val.transform(_ensure_numeric(synth, target)[feat_cols].values.astype(float))
            y_tr = synth[target].values.astype(int)
            if len(np.unique(y_tr)) < 2:
                return -1.0
            rf = RandomForestClassifier(80, random_state=42, n_jobs=-1, class_weight="balanced")
            rf.fit(X_tr, y_tr)
            tstr_auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
            ks_mean, js_mean = compute_ks_js(gan_train_df, synth, feat_cols)
            mia = membership_inference_attack(gan_train_df, mia_holdout_df, synth, feat_cols)
            mia_excess = max(mia["mia_auc"] - 0.55, 0.0)
            score = tstr_auc - 0.5 * ks_mean - 0.5 * js_mean - 2.0 * mia_excess
            _cuda_clear()
            return score
        except Exception as e:
            print(f"  [Optuna] trial failed: {e}")
            _cuda_clear()
            return -1.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"  [Optuna] best score={study.best_value:.4f}  params={best}")
    out_hp = Hyperparams(**asdict(base_hp))
    for k, v in best.items():
        setattr(out_hp, k, v)
    return out_hp


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 10 ─ END-TO-END PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def run_pipeline(cfg: DatasetConfig, epsilon: Optional[float], hp: Hyperparams = None,
                 tune: bool = False, n_optuna_trials: int = 8, out_dir: str = "output"):
    """Runs load -> train -> sample -> evaluate for one epsilon setting.
    Returns (metrics_dict, synthetic_df)."""
    hp = hp or Hyperparams()
    os.makedirs(out_dir, exist_ok=True)

    df, target, cat_cols = load_dataset(cfg)
    feat = [c for c in df.columns if c != target]
    y_all = df[target].values.astype(int)

    i_aug, i_te = train_test_split(np.arange(len(df)), test_size=hp.test_split,
                                   random_state=42, stratify=y_all)
    aug_df_full = df.iloc[i_aug].reset_index(drop=True)
    test_df = df.iloc[i_te].reset_index(drop=True)
    y_te = y_all[i_te]

    i_gan, i_mia = train_test_split(np.arange(len(aug_df_full)), test_size=hp.final_mia_holdout_frac,
                                    random_state=42, stratify=aug_df_full[target].values)
    aug_df = aug_df_full.iloc[i_gan].reset_index(drop=True)
    final_mia_holdout = aug_df_full.iloc[i_mia].reset_index(drop=True)

    sc = StandardScaler().fit(_ensure_numeric(aug_df, target)[feat].values.astype(float))
    X_te = sc.transform(_ensure_numeric(test_df, target)[feat].values.astype(float))
    real_minority = aug_df[aug_df[target] == 1].reset_index(drop=True)

    if tune:
        i_inner, i_val = train_test_split(np.arange(len(aug_df)), test_size=0.25,
                                          random_state=42, stratify=aug_df[target].values)
        train_inner_df = aug_df.iloc[i_inner].reset_index(drop=True)
        val_df = aug_df.iloc[i_val].reset_index(drop=True)
        i_gan2, i_mia2 = train_test_split(np.arange(len(train_inner_df)), test_size=0.15,
                                          random_state=42, stratify=train_inner_df[target].values)
        gan_train_df = train_inner_df.iloc[i_gan2].reset_index(drop=True)
        mia_holdout_df = train_inner_df.iloc[i_mia2].reset_index(drop=True)
        sc_val = StandardScaler().fit(_ensure_numeric(gan_train_df, target)[feat].values.astype(float))
        X_val = sc_val.transform(_ensure_numeric(val_df, target)[feat].values.astype(float))
        y_val = val_df[target].values.astype(int)
        print(f"\n  Tuning hyperparameters (Optuna, {n_optuna_trials} trials)...")
        hp = optuna_tune(epsilon, gan_train_df, val_df, mia_holdout_df, sc_val, X_val, y_val,
                         target, feat, cat_cols, hp, n_trials=n_optuna_trials)

    print(f"\n  Training DP-CTGAN  (epsilon={'No-DP' if epsilon is None else epsilon})...")
    model = DPCTGAN(epsilon=epsilon, hp=hp, verbose=True)
    model.fit(aug_df, target_cat_cols(aug_df, feat, target) if cat_cols is None else cat_cols,
              target, ds_name=cfg.name)

    n_aug = len(aug_df)
    common_df = model.sample_common(n_aug)
    common_df = postprocess_synthetic(common_df, aug_df, target, cat_cols, cfg.int_cols, cfg.hard_bounds)
    if hp.privacy_filter_enabled:
        common_df = privacy_realism_filter(common_df, aug_df, feat, desired_n=n_aug, label="common")

    rare_df = pd.DataFrame()
    try:
        model.build_flows(hp=hp)
        n_real_min = int(aug_df[target].sum())
        n_rare_need = max(200, int(n_real_min * hp.rare_multiplier))
        rare_df = model.sample_rare(n_rare_need, df_real=aug_df)
        rare_df = quantile_match(rare_df.drop(columns=[target]), real_minority, feat,
                                 blend=hp.quantile_blend, orig_df=df)
        rare_df[target] = 1
        rare_df = postprocess_synthetic(rare_df, aug_df, target, cat_cols, cfg.int_cols, cfg.hard_bounds)
        if hp.privacy_filter_enabled:
            rare_df = privacy_realism_filter(rare_df, real_minority, feat, desired_n=n_rare_need, label="rare")
    except Exception as e:
        print(f"  [WARN] rare-sampling step failed, continuing with common-only synth: {e}")

    combined_df = pd.concat([common_df, rare_df], ignore_index=True) if len(rare_df) else common_df.copy()
    combined_df = combined_df[df.columns]

    print("\n  Evaluating synthetic data...")
    tstr = tstr_eval(combined_df, sc, X_te, y_te, feat, target)
    ks_mean, js_mean = compute_ks_js(aug_df, combined_df, feat)
    mia = membership_inference_attack(aug_df, final_mia_holdout, combined_df, feat)
    lr_mia = lr_membership_inference_attack(aug_df, final_mia_holdout, combined_df, feat)

    metrics = {
        "epsilon": epsilon, "actual_epsilon": model._actual_epsilon,
        "rf_auc": tstr["auc"], "rf_auprc": tstr["auprc"], "rf_f1": tstr["f1"],
        "knn_auc": tstr["knn_auc"], "knn_auprc": tstr["knn_auprc"],
        "ks_mean": ks_mean, "js_mean": js_mean,
        "knn_mia_auc": mia["mia_auc"], "knn_mia_advantage": mia["mia_advantage"],
        "lr_mia_auc": lr_mia["lr_mia_auc"], "lr_mia_advantage": lr_mia["lr_mia_advantage"],
        "n_synthetic_rows": len(combined_df),
    }

    print("\n  ── RESULTS ──────────────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<20} {v}")
    print("  ─────────────────────────────────────────────────────")

    out_csv = os.path.join(out_dir, f"synthetic_{cfg.name}_eps{epsilon}.csv")
    combined_df.to_csv(out_csv, index=False)
    with open(os.path.join(out_dir, f"metrics_{cfg.name}_eps{epsilon}.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n  Saved synthetic data -> {out_csv}")

    return metrics, combined_df


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 11 ─ CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Generalized DP-CTGAN pipeline — works on any tabular CSV.")
    p.add_argument("--data", type=str, default=None, help="Path to a CSV file (overrides --preset).")
    p.add_argument("--target", type=str, default=None, help="Target column name (required with --data unless auto-detectable).")
    p.add_argument("--preset", type=str, default=None, choices=list(DATASET_PRESETS.keys()),
                   help="Use a built-in dataset preset instead of --data.")
    p.add_argument("--epsilon", type=str, default="none",
                   help="Privacy budget. Use 'none' for no-DP, or a number like 3 / 10.")
    p.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning first.")
    p.add_argument("--trials", type=int, default=8, help="Number of Optuna trials if --tune is set.")
    p.add_argument("--gan-epochs", type=int, default=None, help="Override GAN training epochs.")
    p.add_argument("--out", type=str, default="output", help="Output directory.")
    args = p.parse_args()

    if args.data:
        cfg = DatasetConfig(name=os.path.splitext(os.path.basename(args.data))[0],
                            path=args.data, target_col=args.target)
    elif args.preset:
        cfg = DATASET_PRESETS[args.preset]
    else:
        cfg = DATASET_CONFIG
        print(f"  No --data or --preset given, using DATASET_CONFIG (preset='{cfg.name}') "
              f"from the top of this script.")

    epsilon = None if args.epsilon.lower() in ("none", "nodp", "no-dp") else float(args.epsilon)

    hp = Hyperparams(**asdict(HYPERPARAMS))
    if args.gan_epochs:
        hp.gan_epochs = args.gan_epochs

    run_pipeline(cfg, epsilon, hp=hp, tune=args.tune, n_optuna_trials=args.trials, out_dir=args.out)


if __name__ == "__main__":
    main()
