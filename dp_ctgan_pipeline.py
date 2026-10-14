#!/usr/bin/env python3
"""
DP-CTGAN (RareFlow-DP) — Differentially-Private Tabular Synthesis Pipeline

One shared engine for every dataset:
  * ModeNormalizer (GMM mode-specific encoding)
  * Conditional generator + discriminator; the discriminator is trained with
    DP-SGD (Opacus, RDP accountant) when an epsilon is given
  * Minority-aware generator objective: adversarial + feature matching (FM)
    + minority moment matching (MM) + minority correlation matching (corr)
    + minority feature matching (MFM), followed by minority fine-tuning
  * Rare-minority sampler in the generator's hidden space:
      Stage 1 pool generation -> Stage 2 StableNVP likelihood-ratio scoring
      -> Stage 3 latent refinement -> Stage 4 RF judge -> Stage 5 greedy
      MaxMin diversity -> Stage 6 partial quantile matching (applied by caller)
  * PrivacyGate near-copy / outlier filter
  * kNN-MIA / LR-MIA evaluation

PRIVACY SCOPE: epsilon bounds the discriminator's DP-SGD updates only. The
generator's FM/MM/corr/MFM losses, fine-tuning, the encoder, the RF judge,
quantile matching and the PrivacyGate read real data outside the accountant,
so the released data is NOT end-to-end (epsilon, delta)-DP. See the
"Privacy Scope" section of the manuscript.

Quick use:
    python dp_ctgan_pipeline.py --preset adult --epsilon 3
    python dp_ctgan_pipeline.py --data yourfile.csv --target label --epsilon none
For the full multi-seed experiment grid used in the paper see
experiments/run_experiments.py and RUNNING_EXPERIMENTS.md.
"""

import os, sys, warnings, time, math, argparse, json, copy
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Callable
from scipy.stats import rankdata, ks_2samp
from scipy.spatial.distance import jensenshannon

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:  # only needed for DP training; lazy so that baseline environments can import this module
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
except Exception:  # pragma: no cover
    PrivacyEngine = ModuleValidator = None

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, f1_score,
    precision_score, recall_score,
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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    """Seed numpy, torch and python RNGs (call once per experiment run)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


#  SECTION 0 ─ DATASET CONFIG

@dataclass
class DatasetConfig:
    name: str
    path: str
    target_col: Optional[str] = None
    target_candidates: List[str] = field(default_factory=list)
    positive_labels: List[str] = field(default_factory=lambda: ["1", "1.0", "true", "yes", ">50k"])
    id_cols: List[str] = field(default_factory=list)       # dropped unconditionally (IDs, free text, leakage)
    cat_cols: Optional[List[str]] = None                   # None = auto-infer
    log1p_cols: List[str] = field(default_factory=list)    # log(1+x) after clipping at 0
    signed_log1p_cols: List[str] = field(default_factory=list)  # sign(x)*log(1+|x|) (keeps negative values)
    int_cols: List[str] = field(default_factory=list)      # rounded after synthesis (raw-scale columns only)
    hard_bounds: Dict[str, tuple] = field(default_factory=dict)
    max_rows: Optional[int] = None                         # subsample majority so len(df)=max_rows; keeps all minority rows
    drop_high_corr_threshold: Optional[float] = 0.95
    custom_clean_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None


@dataclass
class Hyperparams:
    # GAN
    gan_epochs: int = 60
    latent_dim: int = 128
    hidden_dim: int = 256
    batch_size: int = 512
    gen_lr: float = 2e-4
    disc_lr: float = 2e-4
    dp_max_grad_norm: float = 5.0
    delta: Optional[float] = None            # None -> min(1e-5, 1/(10 n_train))
    r1_gamma: float = 5.0
    minority_batch_frac: float = 0.5         # class balance of the real batch used for the generator's matching losses
    lambda_fm: float = 1.0
    lambda_mm: float = 2.0
    lambda_corr: float = 1.0
    lambda_mfm: float = 1.0
    corr_warmup_frac: float = 0.3            # L_corr switched on after this fraction of epochs
    finetune_epochs: int = 20
    finetune_steps_per_epoch: int = 10
    finetune_patience: int = 5
    # flows (Stage 2)
    flow_epochs: int = 60
    flow_layers: int = 6
    flow_hidden: int = 128
    flow_lr: float = 2e-4
    flow_n_samples: int = 8000
    # rare sampler (Stages 1-5)
    rare_multiplier: float = 1.5             # n_rare = max(200, rare_multiplier * n_real_minority)
    rare_pool_size: int = 20000
    rare_max_iters: int = 30
    refine_top_frac: float = 0.25
    refine_steps: int = 50
    refine_lr: float = 0.01
    judge_keep_frac: float = 0.40
    judge_min_prob: float = 0.05
    # Stage 6 + filters
    quantile_blend: float = 0.35
    privacy_filter_enabled: bool = True
    privacy_max_ref: int = 2000
    # evaluation
    knn_k: int = 5
    test_split: float = 0.20
    final_mia_holdout_frac: float = 0.10
    # privacy mode (see apply_privacy_mode): "legacy" = first experiment round (real-data
    # statistics, RF judge and quantile matching outside the accountant); "dp" = every
    # component that reads real data is covered by the DP budget, except the GMM encoder
    privacy_mode: str = "legacy"
    budget_frac_judge: float = 0.10          # share of epsilon for the DP-SGD judge (judge_type="mlp")
    budget_frac_stats: float = 0.10          # share of epsilon for the minority mean/std (Gaussian mechanism)
    judge_type: str = "rf"                   # "rf" (non-private), "mlp" (DP-SGD in dp mode), "disc" (DP discriminator)
    judge_epochs: int = 10
    judge_batch: int = 256
    judge_lr: float = 1e-3
    judge_max_grad_norm: float = 1.0
    mm_noise_ref: float = 0.1                # moment matching is down-weighted when noise/n exceeds this
    finetune_min_mm_weight: float = 0.5      # skip minority fine-tuning when the noisy moments are this unreliable


def apply_privacy_mode(hp: "Hyperparams", mode: str) -> "Hyperparams":
    """Configure the components that read real data.

    legacy: the first experiment round, unchanged.
    dp:     the generator sees real data only through the DP discriminator and through
            minority mean/std released with the Gaussian mechanism; no feature matching,
            correlation matching or quantile matching; the judge is a DP-SGD classifier;
            the PrivacyGate only measures near-copies (filtering would be data-dependent).
    The same configuration is used without DP (epsilon=None), just without noise, so that
    the No-DP column is the same model.
    """
    hp.privacy_mode = mode
    if mode == "dp":
        hp.lambda_fm = hp.lambda_mfm = hp.lambda_corr = 0.0
        hp.quantile_blend = 0.0
        hp.privacy_filter_enabled = False
        hp.judge_type = "mlp"
    elif mode != "legacy":
        raise ValueError(f"unknown privacy mode {mode!r}")
    return hp


DATA_DIR = os.environ.get("DPCTGAN_DATA_DIR", "data")

DATASET_PRESETS: Dict[str, DatasetConfig] = {
    "adult": DatasetConfig(
        name="adult", path=os.path.join(DATA_DIR, "adult.csv"), target_col="income",
        target_candidates=["income", "income_>50K"],
        id_cols=["fnlwgt"],
        cat_cols=["workclass", "education", "marital-status", "occupation",
                  "relationship", "race", "sex", "native-country"],
        log1p_cols=["capital-gain", "capital-loss"],
        int_cols=["age", "education-num", "hours-per-week"],
        hard_bounds={"age": (17, 90), "education-num": (1, 16), "hours-per-week": (1, 99)},
    ),
    "taiwan_credit": DatasetConfig(
        name="taiwan_credit", path=os.path.join(DATA_DIR, "UCI_Credit_Card.csv"),
        target_col="default.payment.next.month",
        target_candidates=["default.payment.next.month", "default", "Y"],
        id_cols=["ID"],
        cat_cols=["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3",
                  "PAY_4", "PAY_5", "PAY_6"],
        log1p_cols=[f"PAY_AMT{i}" for i in range(1, 7)],
        signed_log1p_cols=[f"BILL_AMT{i}" for i in range(1, 7)],
        int_cols=["AGE", "LIMIT_BAL"],
        hard_bounds={"AGE": (21, 79), "LIMIT_BAL": (10_000, 1_000_000)},
    ),
    "credit_fraud": DatasetConfig(
        name="credit_fraud", path=os.path.join(DATA_DIR, "creditcard.csv"), target_col="Class",
        target_candidates=["Class", "class", "target"],
        cat_cols=[],
        log1p_cols=["Amount", "Time"],
        max_rows=100_000,   # all 492 frauds + 99,508 legitimate (as in the paper); set None for all 284,807
    ),
    "mimic3": DatasetConfig(
        name="mimic3", path=os.path.join(DATA_DIR, "mimic3.csv"), target_col="MORTALITY",
        id_cols=["ROW_ID", "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "DIAGNOSIS",
                 "EXPIRE_FLAG", "PRIMARY_ICD9", "DISCHARGE_LOCATION", "DEATHTIME",
                 "DOD", "HOSPITAL_EXPIRE_FLAG", "LANGUAGE", "ETHNICITY", "RELIGION"],
        cat_cols=["GENDER", "ADMISSION_TYPE", "INSURANCE", "MARITAL_STATUS", "FIRST_CAREUNIT"],
        log1p_cols=["HOSPITAL_LOS", "ICU_LOS", "NUM_DIAGNOSES", "NUM_PROCEDURES", "ICU_STAYS"],
        int_cols=["AGE"],
        hard_bounds={"AGE": (18, 90)},
    ),
}

# Per-dataset defaults from the manuscript's hyperparameter table (Appendix A).
HP_PRESETS: Dict[str, Dict] = {
    "adult": dict(gan_epochs=80, dp_max_grad_norm=5.0, r1_gamma=5.0, finetune_patience=8,
                  rare_pool_size=30000, rare_max_iters=30, flow_hidden=128,
                  refine_top_frac=0.25, refine_steps=50, refine_lr=0.010,
                  judge_keep_frac=0.40, judge_min_prob=0.05, quantile_blend=0.35, privacy_max_ref=2000),
    "taiwan_credit": dict(gan_epochs=80, dp_max_grad_norm=5.0, r1_gamma=10.0, finetune_patience=8,
                          rare_pool_size=30000, rare_max_iters=8, flow_hidden=128,
                          refine_top_frac=0.15, refine_steps=40, refine_lr=0.010,
                          judge_keep_frac=0.40, judge_min_prob=0.05, quantile_blend=0.35, privacy_max_ref=8000),
    "credit_fraud": dict(gan_epochs=50, dp_max_grad_norm=1.0, r1_gamma=10.0, finetune_patience=5,
                         rare_pool_size=20000, rare_max_iters=30, flow_hidden=256,
                         refine_top_frac=0.25, refine_steps=50, refine_lr=0.012,
                         judge_keep_frac=0.25, judge_min_prob=0.01, quantile_blend=0.30, privacy_max_ref=2000),
    "mimic3": dict(gan_epochs=100, dp_max_grad_norm=1.0, r1_gamma=10.0, finetune_patience=5,
                   rare_pool_size=20000, rare_max_iters=30, flow_hidden=256,
                   refine_top_frac=0.25, refine_steps=50, refine_lr=0.012,
                   judge_keep_frac=0.25, judge_min_prob=0.01, quantile_blend=0.35, privacy_max_ref=2000),
}


def hyperparams_for(dataset: str, overrides: Optional[Dict] = None) -> Hyperparams:
    hp = Hyperparams(**HP_PRESETS.get(dataset, {}))
    for k, v in (overrides or {}).items():
        if hasattr(hp, k):
            setattr(hp, k, v)
    return hp


DATASET_CONFIG = DATASET_PRESETS["adult"]
HYPERPARAMS = Hyperparams()


#  SECTION 1 ─ GENERIC DATA LOADING

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
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and set(numeric.unique()) <= {0, 1}:
        return numeric.astype(int)
    vc = norm.value_counts()
    minority_label = vc.index[-1] if len(vc) >= 2 else vc.index[0]
    return (norm == minority_label).astype(int)


def load_dataset(cfg: DatasetConfig, verbose: bool = True):
    """Load + clean any CSV according to a DatasetConfig. Returns (df, target, cat_cols)."""
    if not os.path.exists(cfg.path):
        raise FileNotFoundError(f"Dataset not found at '{cfg.path}'. Run data/prepare_public_datasets.py "
                                f"(or data/prepare_mimic3.py) or edit DatasetConfig.path.")

    df = pd.read_csv(cfg.path)
    df.columns = df.columns.str.strip()

    target = cfg.target_col
    if target is None or target not in df.columns:
        for cand in cfg.target_candidates:
            if cand in df.columns:
                target = cand
                break
    if target is None or target not in df.columns:
        raise ValueError(f"Could not find target column. Tried {cfg.target_col!r} and "
                         f"{cfg.target_candidates}. Available columns: {list(df.columns)}")

    drop_cols = [c for c in cfg.id_cols if c in df.columns and c != target]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        if verbose:
            print(f"  [load] dropped ID/leakage columns: {drop_cols}")

    n_before = len(df)
    df = df.replace({"?": np.nan, "": np.nan}).dropna().reset_index(drop=True)
    if verbose and len(df) < n_before:
        print(f"  [load] dropped {n_before - len(df):,} rows with missing values")

    df[target] = _encode_binary_target(df[target], cfg.positive_labels)

    if cfg.max_rows is not None and len(df) > cfg.max_rows:
        pos = df[df[target] == 1]
        neg = df[df[target] == 0].sample(n=max(cfg.max_rows - len(pos), 0), random_state=0)
        df = pd.concat([pos, neg]).sample(frac=1, random_state=0).reset_index(drop=True)
        if verbose:
            print(f"  [load] subsampled majority class to {len(df):,} rows (all {len(pos):,} minority kept)")

    cat_cols = cfg.cat_cols if cfg.cat_cols is not None else _infer_categorical_columns(df, target)
    cat_cols = [c for c in cat_cols if c in df.columns]
    for c in cat_cols:
        df[c] = pd.Categorical(df[c].astype(str).str.strip()).codes

    for c in cfg.log1p_cols:
        if c in df.columns:
            df[c] = np.log1p(pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0))
    for c in cfg.signed_log1p_cols:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").fillna(0)
            df[c] = np.sign(v) * np.log1p(np.abs(v))

    for c in df.columns:
        if c == target:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if cfg.custom_clean_fn is not None:
        df = cfg.custom_clean_fn(df)

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
    if verbose:
        print(f"  [load] {cfg.name}: rows={len(df):,}  positive={n_pos:,} "
              f"({n_pos/len(df)*100:.2f}%)  features={df.shape[1]-1}  cat_cols={len(cat_cols)}")
    return df, target, cat_cols


#  SECTION 2 ─ UTILITIES

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


def greedy_maxmin_pca(arr, k, scores=None, pca_k=32):
    """Greedy farthest-point (MaxMin) selection in a PCA-whitened space.
    Starts from the highest-scoring point; runs on GPU when available."""
    N = len(arr)
    if k >= N:
        return np.arange(N)
    try:
        arr_w = PCA(n_components=min(pca_k, arr.shape[1], N - 1),
                    whiten=True, random_state=42).fit_transform(arr)
    except Exception:
        arr_w = arr
    X = torch.tensor(np.asarray(arr_w, dtype=np.float32), device=DEVICE)
    seed = int(np.argmax(scores)) if scores is not None else 0
    selected = [seed]
    min_d = torch.full((N,), float("inf"), device=DEVICE)
    for _ in range(k - 1):
        d = torch.linalg.vector_norm(X - X[selected[-1]], dim=1)
        min_d = torch.minimum(min_d, d)
        min_d[selected[-1]] = -1.0
        selected.append(int(torch.argmax(min_d).item()))
    return np.array(selected)


def _cuda_clear():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _torch_corr(x):
    xc = x - x.mean(0, keepdim=True)
    cov = xc.T @ xc / max(x.shape[0] - 1, 1)
    std = torch.sqrt(torch.clamp(torch.diag(cov), min=1e-8))
    return cov / (std[:, None] * std[None, :])


#  SECTION 2b ─ DIFFERENTIAL-PRIVACY HELPERS (Gaussian mechanism, RDP composition)

# RDP orders for dp mode. Opacus' default orders stop at 63, which puts a floor of about
# log(1/delta)/62 (~0.22 for delta = 1e-6) on any epsilon it can certify; the budget shares in
# dp mode (e.g. 10% of epsilon = 1) are below that. Larger orders remove the floor.
RDP_ALPHAS = ([1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) + list(range(64, 257, 4))
              + [320, 384, 512, 768, 1024])


def gaussian_noise_multiplier(epsilon, delta):
    """Noise multiplier z (noise std = z * L2-sensitivity) of one Gaussian mechanism that is
    exactly (epsilon, delta)-DP: the analytic Gaussian mechanism (Balle & Wang, ICML 2018, Thm. 8),
        delta(z) = Phi(1/(2z) - eps*z) - exp(eps) * Phi(-1/(2z) - eps*z),
    solved for the smallest z with delta(z) <= delta by bisection."""
    from scipy.stats import norm

    def delta_of(z):
        return norm.cdf(0.5 / z - epsilon * z) - math.exp(epsilon) * norm.cdf(-0.5 / z - epsilon * z)

    lo, hi = 1e-3, 1.0
    while delta_of(hi) > delta:
        hi *= 2.0
        if hi > 1e7:
            raise ValueError("Gaussian mechanism: budget too small")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if delta_of(mid) > delta else (lo, mid)
    return float(hi)


def make_private_dp(pe, module, optimizer, data_loader, epsilon, delta, epochs, max_grad_norm):
    """DP-SGD with the noise calibrated by Opacus' RDP accountant over RDP_ALPHAS (dp mode).
    Same sample rate and number of steps as PrivacyEngine.make_private_with_epsilon, but works
    for small budgets on every Opacus version."""
    from opacus.accountants.utils import get_noise_multiplier
    sigma = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta, sample_rate=1.0 / len(data_loader),
                                 epochs=epochs, accountant="rdp", alphas=RDP_ALPHAS)
    return pe.make_private(module=module, optimizer=optimizer, data_loader=data_loader,
                           noise_multiplier=sigma, max_grad_norm=max_grad_norm)


def rdp_epsilon(pe_or_history, delta):
    """Epsilon over RDP_ALPHAS of a PrivacyEngine or of a list of (noise_multiplier, sample_rate,
    steps) mechanisms (RDP composition)."""
    from opacus.accountants import RDPAccountant
    if hasattr(pe_or_history, "accountant"):
        return float(pe_or_history.accountant.get_epsilon(delta=delta, alphas=RDP_ALPHAS))
    acc = RDPAccountant()
    acc.history = [tuple(h) for h in pe_or_history]
    return float(acc.get_epsilon(delta=delta, alphas=RDP_ALPHAS))


rdp_total_epsilon = rdp_epsilon


def minority_statistics(min_np, cont_idx, n_cont_cols, n_cat_cols, epsilon=None, delta=None, rng=None):
    """Per-coordinate mean and std of the encoded minority records.

    epsilon=None: exact statistics (as in the first experiment round).
    epsilon given: one Gaussian mechanism on the vector of sums
        [sum_i x_i (all encoded coordinates), sum_i a_i^2 (continuous scalars), c * count].
    Every encoded record has bounded norm by construction (continuous scalars are clipped to
    +-0.99 and each column has exactly one active one-hot entry), so the L2 sensitivity under
    add/remove-one adjacency is data-independent:
        B^2 = n_cont * (0.99^2 + 1 + 0.99^4) + n_cat,   sensitivity = sqrt(B^2 + c^2), c = B.
    Means and stds are post-processed from the noisy sums (clipped to their valid ranges).
    Returns (mean, std, info)."""
    d = min_np.shape[1]
    cont_idx = np.asarray(cont_idx, dtype=int)
    onehot = np.ones(d, dtype=bool)
    onehot[cont_idx] = False
    n = len(min_np)
    if epsilon is None:
        return min_np.mean(0), min_np.std(0) + 1e-6, {"n_hat": float(n), "rel_noise": 0.0}
    rng = rng or np.random.default_rng()
    B = math.sqrt(n_cont_cols * (0.99 ** 2 + 1.0 + 0.99 ** 4) + n_cat_cols)
    c = B
    sens = math.sqrt(B ** 2 + c ** 2)
    z = gaussian_noise_multiplier(epsilon, delta)
    sd = z * sens
    s1 = min_np.sum(0) + rng.normal(0, sd, d)
    s2 = (min_np[:, cont_idx] ** 2).sum(0) + rng.normal(0, sd, len(cont_idx))
    n_hat = max((c * n + rng.normal(0, sd)) / c, 20.0)
    mean = s1 / n_hat
    mean[onehot] = np.clip(mean[onehot], 0.0, 1.0)
    mean[cont_idx] = np.clip(mean[cont_idx], -0.99, 0.99)
    std = np.sqrt(np.clip(mean * (1 - mean), 0.0, None)) + 1e-6
    var_c = np.clip(s2 / n_hat - mean[cont_idx] ** 2, 1e-4, 1.0)
    std[cont_idx] = np.sqrt(var_c)
    info = {"epsilon": float(epsilon), "delta": float(delta), "noise_multiplier": z, "sensitivity": sens,
            "n_hat": float(n_hat), "rel_noise": float(sd / n_hat)}
    return mean, std, info


#  SECTION 3 ─ DISTRIBUTION METRICS + MIA + PRIVACY GATE

def compute_ks_js(real_df, synth_df, feat_cols, n_bins=50):
    """Mean KS statistic and mean Jensen-Shannon DISTANCE (scipy, base e) over features."""
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
    nn_ = NearestNeighbors(n_neighbors=kk, n_jobs=-1).fit(X_synth)
    d_mem, _ = nn_.kneighbors(X_mem)
    d_non, _ = nn_.kneighbors(X_non)
    y_true = np.concatenate([np.ones(len(d_mem)), np.zeros(len(d_non))])
    y_score = np.concatenate([-d_mem.mean(1), -d_non.mean(1)])
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = 0.5
    return {"mia_auc": auc, "mia_advantage": 2 * (auc - 0.5)}


def lr_membership_inference_attack(train_df, holdout_df, synth_df, feat_cols,
                                   n_eval=2000, n_shadow=5, fpr_thresholds=(0.01, 0.05, 0.10),
                                   seed=42):
    """Classifier-confidence attack (LR-MIA). Shadow labels come only from synth_df
    (its label column if present, else a PCA median split), never from membership."""
    rng = np.random.RandomState(seed)
    n_eval = max(50, min(n_eval, len(train_df), len(holdout_df)))
    mem_idx = rng.choice(len(train_df), n_eval, replace=False)
    non_idx = rng.choice(len(holdout_df), n_eval, replace=False)
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
        idx = rng.choice(len(X_synth_sc), n_sh, replace=False)
        y_sh = synth_labels[idx]
        if len(np.unique(y_sh)) < 2:
            continue
        try:
            clf = LogisticRegression(max_iter=300, solver="saga", C=1.0, random_state=sh * 7 + 42)
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
                           min_q=0.03, max_q=0.995, max_ref=2000, label="", verbose=True,
                           return_stats=False, audit_only=False):
    """PrivacyGate. Removes near-copies (closer to the reference data than the 3% quantile
    of real-to-real kNN distances) and outliers (farther than 1.25x the 99.5% quantile).
    If fewer than desired_n survive, the shortfall is refilled ONLY from outlier rejects;
    near-copies are never re-admitted.
    audit_only=True measures the same rates but returns synth_df unchanged (dp mode: a
    data-dependent filter would not be covered by the privacy budget)."""
    stats = {"n_in": int(len(synth_df)), "keep_rate": 1.0, "near_copy_rate": 0.0,
             "outlier_rate": 0.0, "n_out": int(len(synth_df))}
    if len(synth_df) == 0:
        return (synth_df, stats) if return_stats else synth_df
    use_cols = [c for c in feat_cols if c in ref_df.columns and c in synth_df.columns]
    if not use_cols:
        out = synth_df.head(desired_n) if desired_n else synth_df
        return (out, stats) if return_stats else out
    ref_s = ref_df.sample(n=min(len(ref_df), max_ref), random_state=42)
    X_ref = _ensure_numeric(ref_s[use_cols], "__none__")[use_cols].values.astype(float)
    X_syn = _ensure_numeric(synth_df[use_cols], "__none__")[use_cols].values.astype(float)
    sc = StandardScaler().fit(X_ref)
    X_ref, X_syn = sc.transform(X_ref), sc.transform(X_syn)
    k_ref = min(k + 1, len(X_ref))
    nn_ref = NearestNeighbors(n_neighbors=k_ref, n_jobs=-1).fit(X_ref)
    d_ref, _ = nn_ref.kneighbors(X_ref)
    ref_dist = d_ref[:, 1:].mean(1) if d_ref.shape[1] > 1 else d_ref[:, 0]
    d_syn, _ = NearestNeighbors(n_neighbors=min(k, len(X_ref)), n_jobs=-1).fit(X_ref).kneighbors(X_syn)
    syn_dist = d_syn.mean(1)
    lower = max(float(np.quantile(ref_dist, min_q)), 1e-6)
    upper = max(float(np.quantile(ref_dist, max_q)) * 1.25, lower * 1.5)
    near = syn_dist < lower
    far = syn_dist > upper
    keep_mask = ~(near | far)
    df = synth_df.reset_index(drop=True)
    safe = df.loc[keep_mask]
    if audit_only:
        stats.update({"keep_rate": float(keep_mask.mean()), "near_copy_rate": float(near.mean()),
                      "outlier_rate": float(far.mean()), "n_out": int(len(df)), "audit_only": True,
                      "lower_bound": lower, "upper_bound": upper, "ref_size": int(len(X_ref))})
        if verbose:
            print(f"  [PrivacyGate:{label} audit] near-copy={near.sum():,} outlier={far.sum():,} of {len(df):,}")
        return (df, stats) if return_stats else df
    if desired_n is None:
        out = safe
    elif len(safe) >= desired_n:
        out = safe.sample(n=desired_n, random_state=42)
    else:
        outliers = df.loc[far]
        n_fill = min(desired_n - len(safe), len(outliers))
        out = pd.concat([safe, outliers.sample(n=n_fill, random_state=42)]) if n_fill > 0 else safe
    n = len(df)
    stats.update({"keep_rate": float(keep_mask.mean()), "near_copy_rate": float(near.mean()),
                  "outlier_rate": float(far.mean()), "n_out": int(len(out)),
                  "lower_bound": lower, "upper_bound": upper, "ref_size": int(len(X_ref))})
    if verbose:
        print(f"  [PrivacyGate:{label}] keep={keep_mask.sum():,}/{n:,} near-copy={near.sum():,} "
              f"outlier={far.sum():,} return={len(out):,}")
    out = out.reset_index(drop=True)
    return (out, stats) if return_stats else out


#  SECTION 4 ─ EVALUATION (RF + KNN downstream classifiers, threshold 0.5)
#  The paper's evaluation (validation-selected thresholds etc.) lives in
#  experiments/evaluation.py; these simple versions are kept for the CLI.

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


#  SECTION 5 ─ QUANTILE MATCHING + POSTPROCESSING

def quantile_match(synth_df, ref_df, feat_cols, blend, orig_df=None):
    """Stage 6: blend each column with the reference quantile at the same rank.
    Pass only CONTINUOUS columns in feat_cols, and the TRAINING data as orig_df
    (bounds) — never the full dataset, which would leak the test range."""
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


#  SECTION 6 ─ MODE NORMALIZER (vectorised)

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
            nv = max(int(df[c].max()) + 1, int(df[c].nunique()), 1)
            self.cat_nvals_[c] = nv
            self._slices[c] = slice(idx, idx + nv)
            idx += nv
        self.out_dim_ = idx
        return self

    @property
    def cont_scalar_idx(self):
        return [self._slices[c].start for c in self.cont_cols_]

    def transform(self, df):
        n = len(df)
        rows = np.arange(n)
        out = np.zeros((n, self.out_dim_), dtype=np.float32)
        for c in self.cont_cols_:
            v = df[c].values.reshape(-1, 1).astype(float)
            gm = self.gmms_[c]
            modes = gm.predict(v)
            means = gm.means_.flatten()[modes]
            stds = np.sqrt(gm.covariances_.reshape(len(gm.means_), -1)[:, 0])[modes]
            sl = self._slices[c]
            out[:, sl.start] = np.clip((v.flatten() - means) / (4 * stds + 1e-8), -0.99, 0.99)
            out[rows, sl.start + 1 + modes] = 1.
        for c in self.cat_cols_:
            vals = df[c].values.astype(int)
            nv = self.cat_nvals_[c]
            sl = self._slices[c]
            out[rows, sl.start + (vals % nv)] = 1.
        return out

    def inverse_transform(self, data, orig_df):
        data = np.asarray(data)
        res = {}
        for c in self.cont_cols_:
            sl, gm = self._slices[c], self.gmms_[c]
            m = np.argmax(data[:, sl.start + 1:sl.stop], axis=1)
            stds = np.sqrt(gm.covariances_.reshape(len(gm.means_), -1)[:, 0])[m]
            means = gm.means_.flatten()[m]
            val = data[:, sl.start].astype(float) * 4. * stds + means
            res[c] = np.clip(val, float(orig_df[c].min()), float(orig_df[c].max()))
        for c in self.cat_cols_:
            sl = self._slices[c]
            res[c] = np.argmax(data[:, sl], axis=1) % self.cat_nvals_[c]
        return pd.DataFrame(res, columns=self.cont_cols_ + self.cat_cols_)


#  SECTION 7 ─ GAN NETWORKS + STABLE NVP FLOW

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
    """RealNVP with tanh-bounded log-scales. Dimensions are reversed between
    coupling layers so that every coordinate is transformed."""
    def __init__(self, dim, n_layers=6, hidden=128, scale_max=2.0):
        super().__init__()
        self.pad = dim % 2
        self.dim = dim + self.pad
        self.layers = nn.ModuleList([StableAffineCoupling(self.dim, hidden, scale_max) for _ in range(n_layers)])

    def log_prob(self, x):
        if self.pad:
            x = torch.cat([x, torch.zeros(x.size(0), self.pad, device=x.device)], 1)
        ld = torch.zeros(x.size(0), device=x.device)
        for i, layer in enumerate(self.layers):
            x, d = layer(x)
            ld = ld + d
            if i < len(self.layers) - 1:
                x = torch.flip(x, dims=[1])
        return (-0.5 * (x.pow(2).sum(1) + self.dim * math.log(2 * math.pi)) + ld).clamp(-80., 80.)


def _train_flow(flow, data_tensor, epochs, lr, device):
    flow = flow.to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=lr * 0.1)
    bs = min(256, len(data_tensor))
    dl = DataLoader(TensorDataset(data_tensor), batch_size=bs, shuffle=True, drop_last=True)
    flow.train()
    for _ in range(epochs):
        for (x,) in dl:
            loss = -flow.log_prob(x.to(device)).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            opt.step()
        sched.step()
    flow.eval()
    for p in flow.parameters():
        p.requires_grad_(False)


#  SECTION 7b ─ JUDGES FOR STAGE 4
#  A judge scores decoded minority candidates; Stage 4 keeps the top judge_keep_frac.
#  score(raw, decoded) takes the generator output (encoded space) and its decoded frame.

class RFJudge:
    """Random forest on the real training data (first experiment round). NOT private."""
    kind, private, has_prob = "rf", False, True

    def __init__(self, target_col, feat_cols):
        self.target_col, self.feat_cols = target_col, feat_cols
        self.info = {"kind": self.kind, "private": False}

    def fit(self, df_real):
        X = _ensure_numeric(df_real, self.target_col)[self.feat_cols].values.astype(float)
        y = df_real[self.target_col].values.astype(int)
        self.clf = RandomForestClassifier(150, random_state=42, n_jobs=-1, class_weight="balanced").fit(X, y)
        return self

    def score(self, raw, decoded):
        return self.clf.predict_proba(decoded[self.feat_cols].values.astype(float))[:, 1]


class MLPJudge:
    """MLP minority classifier on the GMM-encoded records, trained with DP-SGD (Opacus, RDP
    accountant) when epsilon is given, otherwise with plain SGD (same architecture)."""
    kind, has_prob = "mlp", True

    def __init__(self, norm, feat_cols, hp: Hyperparams):
        self.norm, self.feat_cols, self.hp = norm, feat_cols, hp
        self.private = False
        self.info = {"kind": self.kind, "private": False}

    def fit(self, df_feat, y, epsilon=None, delta=None, pos_weight=1.0):
        hp = self.hp
        X = torch.tensor(self.norm.transform(df_feat[self.feat_cols]), dtype=torch.float32)
        Y = torch.tensor(np.asarray(y, dtype=np.float32)).unsqueeze(1)
        net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                            nn.Linear(64, 1)).to(DEVICE)
        opt = optim.Adam(net.parameters(), lr=hp.judge_lr)
        loader = DataLoader(TensorDataset(X, Y), batch_size=min(hp.judge_batch, len(X)), shuffle=True)
        loss_f = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], device=DEVICE))
        pe = None
        if epsilon is not None:
            pe = PrivacyEngine(accountant="rdp")
            net, opt, loader = make_private_dp(pe, net, opt, loader, epsilon, delta, hp.judge_epochs,
                                               hp.judge_max_grad_norm)
        net.train()
        for _ in range(hp.judge_epochs):
            for xb, yb in loader:
                if len(xb) == 0:
                    continue
                loss = loss_f(net(xb.to(DEVICE)), yb.to(DEVICE))
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        self.net = net
        self.private = epsilon is not None
        self.info = {"kind": self.kind, "private": self.private, "pos_weight": float(pos_weight),
                     "epochs": hp.judge_epochs}
        if pe is not None:
            self.history = [tuple(h) for h in pe.accountant.history]
            self.info.update({"epsilon_target": float(epsilon), "delta": float(delta),
                              "noise_multiplier": float(opt.noise_multiplier),
                              "sample_rate": float(getattr(loader, "sample_rate", np.nan)),
                              "steps": int(sum(h[2] for h in self.history)),
                              "achieved_epsilon": rdp_epsilon(pe, delta)})
        return self

    def score(self, raw, decoded):
        X = torch.tensor(self.norm.transform(decoded[self.feat_cols]), dtype=torch.float32)
        out = []
        with torch.no_grad():
            for s in range(0, len(X), 8192):
                out.append(torch.sigmoid(self.net(X[s:s + 8192].to(DEVICE))).squeeze(1).cpu())
        return torch.cat(out).numpy()


class DiscJudge:
    """Scores candidates with the trained (DP) discriminator:
    D(x, c_min) - D(x, c_maj), i.e. how much more x looks like a real minority record than a
    real majority record. Post-processing of the DP discriminator: no extra privacy budget."""
    kind, private, has_prob = "disc", True, False

    def __init__(self, model):
        self.model = model
        self.info = {"kind": self.kind, "private": model.use_dp, "extra_epsilon": 0.0}
        self.private = model.use_dp

    def score(self, raw, decoded):
        D, m = self.model._raw_D, self.model
        D.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(raw), 8192):
                x = torch.tensor(raw[s:s + 8192], dtype=torch.float32, device=DEVICE)
                n = len(x)
                out.append((D(torch.cat([x, m._cond(n, 'minority')], 1))
                            - D(torch.cat([x, m._cond(n, 'majority')], 1))).squeeze(1).cpu())
        return torch.cat(out).numpy()


#  SECTION 8 ─ DP-CTGAN ENGINE

class DPCTGAN:
    """Conditional tabular GAN with a DP-SGD discriminator, minority-aware
    generator objective, and a flow-guided rare-minority sampler."""

    def __init__(self, epsilon=None, delta=None, hp: Hyperparams = None, verbose=True):
        hp = hp or Hyperparams()
        self.hp = hp
        self.epsilon = epsilon
        self.use_dp = epsilon is not None
        self.delta = delta if delta is not None else hp.delta
        self.epochs = hp.gan_epochs
        self.batch_size = hp.batch_size
        self.latent_dim = hp.latent_dim
        self.hidden_dims = (hp.hidden_dim, hp.hidden_dim)
        self.max_grad_norm = hp.dp_max_grad_norm
        self.r1_gamma = hp.r1_gamma if not self.use_dp else 0.0
        self.verbose = verbose
        self._fitted = False
        self.flow_min = self.flow_maj = None
        self._minority_rate = 0.0
        self._actual_epsilon = float('inf')
        self.epsilon_total = float('inf')     # all components (basic composition); = _actual_epsilon in legacy mode
        self.dp_info = {}
        self.budget = {}
        self.judge = None
        self._mm_weight = 1.0

    # -- helpers --------------------------------------------------------
    def _cond(self, n, mode='majority'):
        c = torch.zeros(n, 2, device=DEVICE)
        c[:, 1 if mode == 'minority' else 0] = 1.
        return c

    @staticmethod
    def _onehot(y):
        return F.one_hot(y.long(), 2).float()

    def _balanced_batch(self, bs):
        n_min = max(1, int(bs * self.hp.minority_batch_frac)) if len(self._minority_idx) else 0
        n_maj = bs - n_min
        idx = np.concatenate([np.random.choice(self._minority_idx, n_min, replace=True) if n_min else np.array([], int),
                              np.random.choice(self._majority_idx, n_maj, replace=True)])
        return self._data_tensor[idx].to(DEVICE), self._label_tensor[idx].to(DEVICE)

    # -- training -------------------------------------------------------
    def fit(self, df, discrete_columns, target_col, ds_name=""):
        t0 = time.time()
        hp = self.hp
        self._target_col = target_col
        feat_df = df.drop(columns=[target_col])
        self._feat_cols = list(feat_df.columns)
        self._orig_df = feat_df.copy()
        self._norm = ModeNormalizer().fit(feat_df, [c for c in discrete_columns if c != target_col])
        data_np = self._norm.transform(feat_df)
        data_dim = data_np.shape[1]
        n = len(df)
        if self.delta is None:
            self.delta = min(1e-5, 1.0 / (10 * n))
        dp_mode = hp.privacy_mode == "dp"

        # ---- privacy budget split (dp mode): discriminator / minority statistics / judge ----
        f_judge = hp.budget_frac_judge if (dp_mode and hp.judge_type == "mlp") else 0.0
        f_stats = hp.budget_frac_stats if dp_mode else 0.0
        if self.use_dp and dp_mode:
            self.eps_gan = self.epsilon * (1.0 - f_judge - f_stats)
            self.delta_gan = self.delta / 3.0
            self.budget = {"epsilon_target": float(self.epsilon), "delta": float(self.delta),
                           "epsilon_gan": float(self.eps_gan), "epsilon_stats": float(self.epsilon * f_stats),
                           "epsilon_judge": float(self.epsilon * f_judge), "delta_each": float(self.delta / 3.0)}
        else:
            self.eps_gan, self.delta_gan = self.epsilon, self.delta

        labels = df[target_col].values.astype(int)
        self._minority_idx = np.where(labels == 1)[0]
        self._majority_idx = np.where(labels == 0)[0]
        self._minority_rate = len(self._minority_idx) / max(n, 1)
        min_np = data_np[self._minority_idx] if len(self._minority_idx) else data_np[:1]
        self._cont_idx = self._norm.cont_scalar_idx
        if self.use_dp and dp_mode:
            mean, std, self.stats_info = minority_statistics(
                min_np, self._cont_idx, len(self._norm.cont_cols_), len(self._norm.cat_cols_),
                epsilon=self.epsilon * f_stats, delta=self.delta / 3.0,
                rng=np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item())))
            # down-weight moment matching when the released moments are dominated by noise
            # (a function of released quantities only, i.e. post-processing)
            self._mm_weight = 1.0 / (1.0 + (self.stats_info["rel_noise"] / hp.mm_noise_ref) ** 2)
            self._minority_rate = min(max(self.stats_info["n_hat"] / n, 1.0 / n), 0.5)
        else:
            mean, std, self.stats_info = minority_statistics(min_np, self._cont_idx, 0, 0)
        self.stats_info["mm_weight"] = float(self._mm_weight)
        self._min_mean = torch.tensor(mean, dtype=torch.float32, device=DEVICE)
        self._min_std = torch.tensor(std, dtype=torch.float32, device=DEVICE)
        self._real_min_corr = None
        if hp.lambda_corr > 0 and len(self._cont_idx) >= 2 and len(self._minority_idx) >= 3:
            cm = np.nan_to_num(np.corrcoef(min_np[:, self._cont_idx].T), nan=0.0)
            self._real_min_corr = torch.tensor(cm, dtype=torch.float32, device=DEVICE)

        G = Generator(self.latent_dim, 2, data_dim, self.hidden_dims).to(DEVICE)
        D = DiscriminatorNet(data_dim + 2, self.hidden_dims).to(DEVICE)
        if self.use_dp:
            D = ModuleValidator.fix(D)

        opt_G = optim.Adam(G.parameters(), lr=hp.gen_lr, betas=(0.5, 0.9))
        opt_D = optim.Adam(D.parameters(), lr=hp.disc_lr, betas=(0.5, 0.9))
        bce = nn.BCEWithLogitsLoss()
        self._data_tensor = torch.tensor(data_np, dtype=torch.float32)
        self._label_tensor = torch.tensor(labels, dtype=torch.long)
        bs_loader = min(self.batch_size, max(2, n // 2))
        loader = DataLoader(TensorDataset(self._data_tensor, self._label_tensor),
                            batch_size=bs_loader, shuffle=True, drop_last=True)

        pe = None
        if self.use_dp:
            pe = PrivacyEngine(accountant="rdp")
            if dp_mode:
                D, opt_D, loader = make_private_dp(pe, D, opt_D, loader, self.eps_gan, self.delta_gan,
                                                   self.epochs, self.max_grad_norm)
            else:   # first experiment round, unchanged
                D, opt_D, loader = pe.make_private_with_epsilon(
                    module=D, optimizer=opt_D, data_loader=loader,
                    target_epsilon=self.eps_gan, target_delta=self.delta_gan,
                    epochs=self.epochs, max_grad_norm=self.max_grad_norm)
            D.grad_accumulation_allowed = True
            raw_D = D._module
            self.dp_info = {"noise_multiplier": float(opt_D.noise_multiplier),
                            "sample_rate": float(getattr(loader, "sample_rate", bs_loader / n)),
                            "steps_planned": int(self.epochs * len(loader)),
                            "max_grad_norm": float(self.max_grad_norm), "delta": float(self.delta_gan)}
        else:
            raw_D = D
        eps_of = (lambda d: rdp_epsilon(pe, d)) if dp_mode else (lambda d: pe.get_epsilon(d))

        self._G, self._D, self._raw_D = G, D, raw_D
        need_real = hp.lambda_fm > 0 or hp.lambda_mfm > 0   # generator losses on real batches (legacy only)
        dp_tag = f"eps={self.epsilon}" if self.use_dp else "No-DP"
        if self.verbose:
            extra = f" sigma={self.dp_info['noise_multiplier']:.3f}" if self.use_dp else ""
            print(f"  DPCTGAN [{dp_tag}] [{ds_name}] epochs={self.epochs} batch={bs_loader} "
                  f"minority_rate={self._minority_rate:.4f} delta={self.delta:.2e}{extra}")

        warmup_epoch = int(self.epochs * hp.corr_warmup_frac)
        log_every = max(1, self.epochs // 5)
        steps_done = 0
        for epoch in range(1, self.epochs + 1):
            G.train(); D.train()
            for xb, yb in loader:
                bs_d = xb.size(0)
                if bs_d < 2:
                    continue
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                cond_d = self._onehot(yb)
                with torch.no_grad():
                    fake_d = G(torch.randn(bs_d, self.latent_dim, device=DEVICE), cond_d)
                if self.use_dp or self.r1_gamma <= 0:
                    both = D(torch.cat([torch.cat([xb, cond_d], 1), torch.cat([fake_d, cond_d], 1)], 0))
                    loss_D = (bce(both[:bs_d], torch.full((bs_d, 1), 0.9, device=DEVICE))
                              + bce(both[bs_d:], torch.zeros(bs_d, 1, device=DEVICE))) * 0.5
                else:
                    real_gp = xb.clone().detach().requires_grad_(True)
                    d_real = D(torch.cat([real_gp, cond_d], 1))
                    grad = torch.autograd.grad(d_real.sum(), real_gp, create_graph=True, retain_graph=True)[0]
                    r1 = (self.r1_gamma / 2.0) * grad.pow(2).sum(1).mean()
                    d_fake = D(torch.cat([fake_d, cond_d], 1))
                    loss_D = (bce(d_real, torch.full((bs_d, 1), 0.9, device=DEVICE))
                              + bce(d_fake, torch.zeros(bs_d, 1, device=DEVICE))) * 0.5 + r1
                opt_D.zero_grad(); loss_D.backward(); opt_D.step()
                steps_done += 1

                # ---- generator step ----
                # legacy: the matching losses read a real (class-balanced) batch outside the DP
                # accountant. dp mode: no real batch; conditions are drawn with a fixed public rate.
                bs = self.batch_size
                zero = torch.zeros((), device=DEVICE)
                if need_real:
                    rb, rb_y = self._balanced_batch(bs)
                else:
                    rb_y = (torch.rand(bs, device=DEVICE) < hp.minority_batch_frac).long()
                cond_g = self._onehot(rb_y)
                fake_g = G(torch.randn(bs, self.latent_dim, device=DEVICE), cond_g)
                f_in = torch.cat([fake_g, cond_g], 1)
                adv = bce(raw_D(f_in), torch.ones(bs, 1, device=DEVICE))
                fm = mfm = zero
                if need_real:
                    feat_fake = raw_D.get_features(f_in)
                    with torch.no_grad():
                        feat_real = raw_D.get_features(torch.cat([rb, cond_g], 1))
                    fm = F.mse_loss(feat_fake.mean(0), feat_real.mean(0))
                    mmask = rb_y == 1
                    mfm = (F.l1_loss(feat_fake[mmask].mean(0), feat_real[mmask].mean(0))
                           if int(mmask.sum()) > 1 else zero)

                mm_bs = min(256, bs)
                fake_min = G(torch.randn(mm_bs, self.latent_dim, device=DEVICE), self._cond(mm_bs, 'minority'))
                mm_loss = (F.l1_loss(fake_min.mean(0), self._min_mean) + F.l1_loss(fake_min.std(0), self._min_std))
                corr_loss = zero
                if self._real_min_corr is not None and epoch > warmup_epoch and hp.lambda_corr > 0:
                    corr_loss = F.mse_loss(_torch_corr(fake_min[:, self._cont_idx]), self._real_min_corr)

                loss_G = (adv + hp.lambda_fm * fm + hp.lambda_mm * self._mm_weight * mm_loss
                          + hp.lambda_corr * corr_loss + hp.lambda_mfm * mfm)
                opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            if self.verbose and epoch % log_every == 0:
                eps_s = f"  eps={eps_of(self.delta_gan):.3f}" if self.use_dp else ""
                print(f"  ep {epoch:3d}/{self.epochs}{eps_s}")

        self._actual_epsilon = eps_of(self.delta_gan) if self.use_dp else float('inf')
        if self.use_dp:
            self.dp_info.update({"steps_done": int(steps_done), "achieved_epsilon": float(self._actual_epsilon)})
            self._gan_history = [tuple(h) for h in pe.accountant.history]
        self.epsilon_total = self._actual_epsilon

        # ---- minority fine-tuning of G with early stopping (D frozen) ----
        # skipped when the (noisy) moments are too unreliable to fine-tune towards
        if hp.finetune_epochs > 0 and self._mm_weight >= hp.finetune_min_mm_weight:
            G.train()
            opt_ft = optim.Adam(G.parameters(), lr=max(hp.gen_lr * 0.1, 1e-6))
            best, bad, best_state = float("inf"), 0, copy.deepcopy(G.state_dict())
            for _ in range(hp.finetune_epochs):
                ep_loss = []
                for _ in range(hp.finetune_steps_per_epoch):
                    fake_min = G(torch.randn(256, self.latent_dim, device=DEVICE), self._cond(256, 'minority'))
                    ft_loss = (F.l1_loss(fake_min.mean(0), self._min_mean)
                               + 0.5 * F.l1_loss(fake_min.std(0), self._min_std))
                    opt_ft.zero_grad(); ft_loss.backward(); opt_ft.step()
                    ep_loss.append(float(ft_loss.item()))
                m = float(np.mean(ep_loss))
                if m < best - 1e-5:
                    best, bad, best_state = m, 0, copy.deepcopy(G.state_dict())
                else:
                    bad += 1
                    if bad >= hp.finetune_patience:
                        break
            G.load_state_dict(best_state)
        G.eval()
        for p in G.parameters():
            p.requires_grad_(False)

        # ---- judge for Stage 4 (dp mode; legacy mode builds its RF inside sample_rare) ----
        if dp_mode:
            if hp.judge_type == "mlp":
                eps_j = self.epsilon * f_judge if self.use_dp else None
                n_hat = self.stats_info["n_hat"]
                pos_weight = max(n - n_hat, 1.0) / max(n_hat, 1.0)
                self.judge = MLPJudge(self._norm, self._feat_cols, hp).fit(
                    feat_df, labels, epsilon=eps_j, delta=self.delta / 3.0 if self.use_dp else None,
                    pos_weight=pos_weight)
            elif hp.judge_type == "disc":
                self.judge = DiscJudge(self)
            elif hp.judge_type == "rf":
                self.judge = RFJudge(target_col, self._feat_cols).fit(df)
            else:
                raise ValueError(f"unknown judge_type {hp.judge_type!r}")

        # ---- total privacy cost (dp mode) ----
        if self.use_dp and dp_mode:
            hist = list(self._gan_history) + [(self.stats_info["noise_multiplier"], 1.0, 1)]
            eps_parts = [self._actual_epsilon, self.stats_info["epsilon"]]
            if isinstance(self.judge, MLPJudge) and self.judge.private:
                hist += self.judge.history
                eps_parts.append(self.judge.info["achieved_epsilon"])
            self.epsilon_total = float(sum(eps_parts))
            self.budget.update({"epsilon_gan_achieved": float(self._actual_epsilon),
                                "epsilon_total_basic": self.epsilon_total,
                                "epsilon_total_rdp": rdp_total_epsilon(hist, self.delta),
                                "stats_noise_multiplier": self.stats_info["noise_multiplier"],
                                "stats_rel_noise": self.stats_info["rel_noise"],
                                "mm_weight": float(self._mm_weight),
                                "n_minority_hat": self.stats_info["n_hat"],
                                "finetuned": bool(hp.finetune_epochs > 0 and
                                                  self._mm_weight >= hp.finetune_min_mm_weight)})
            if isinstance(self.judge, MLPJudge) and self.judge.private:
                self.budget.update({"judge_noise_multiplier": self.judge.info["noise_multiplier"],
                                    "judge_steps": self.judge.info["steps"],
                                    "epsilon_judge_achieved": self.judge.info["achieved_epsilon"]})
            if self.verbose:
                print(f"  privacy: eps_gan={self._actual_epsilon:.3f} eps_stats={self.stats_info['epsilon']:.3f} "
                      f"eps_judge={eps_parts[2] if len(eps_parts) > 2 else 0:.3f} -> total "
                      f"{self.epsilon_total:.3f} (basic), {self.budget['epsilon_total_rdp']:.3f} (RDP); "
                      f"moment noise/n={self.stats_info['rel_noise']:.3f}, mm_weight={self._mm_weight:.2f}")

        self._fitted = True
        self.train_seconds = time.time() - t0
        if self.verbose:
            print(f"  Training done  {self.train_seconds:.1f}s  "
                  f"eps={'inf' if not self.use_dp else f'{self.epsilon_total:.3f}'}")
        return self

    # -- sampling -------------------------------------------------------
    def _decode(self, raw):
        synth = self._norm.inverse_transform(raw, self._orig_df)
        return synth[self._feat_cols].reset_index(drop=True)

    def _sample_raw(self, n, mode, batch=2048):
        rows = []
        with torch.no_grad():
            for s in range(0, n, batch):
                nb = min(batch, n - s)
                z = torch.randn(nb, self.latent_dim, device=DEVICE)
                rows.append(self._G(z, self._cond(nb, mode)).cpu().numpy())
        raw = np.vstack(rows) if rows else np.zeros((0, self._norm.out_dim_), dtype=np.float32)
        return self._decode(raw), raw

    def sample_common(self, n):
        """Conditional sampling at the empirical class ratio."""
        n_min = max(1, int(round(n * self._minority_rate)))
        df_maj, _ = self._sample_raw(n - n_min, 'majority')
        df_min, _ = self._sample_raw(n_min, 'minority')
        df_maj[self._target_col] = 0
        df_min[self._target_col] = 1
        return pd.concat([df_maj, df_min], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    def sample_minority_plain(self, n):
        """Ablation: n minority samples drawn directly from G(z, c_min)."""
        df_min, _ = self._sample_raw(n, 'minority')
        df_min[self._target_col] = 1
        return df_min

    def _hidden(self, z, mode='minority', batch=4096):
        hs = []
        with torch.no_grad():
            for s in range(0, len(z), batch):
                zb = z[s:s + batch].to(DEVICE)
                hs.append(self._G.shared(torch.cat([zb, self._cond(len(zb), mode)], 1)))
        return torch.cat(hs)

    def build_flows(self, hp: Hyperparams = None):
        """Stage 2 setup: PCA(32, whitened) + standardisation of hidden states, then
        one flow on minority-conditioned and one on majority-conditioned states."""
        assert self._fitted
        hp = hp or self.hp
        n_samp = hp.flow_n_samples
        h_min = self._hidden(torch.randn(n_samp, self.latent_dim), 'minority').cpu().numpy()
        h_maj = self._hidden(torch.randn(n_samp, self.latent_dim), 'majority').cpu().numpy()
        h_all = np.concatenate([h_min, h_maj])
        pca_k = min(32, h_all.shape[1], len(h_all) - 1)
        self._flow_pca = PCA(n_components=pca_k, whiten=True, random_state=42).fit(h_all)
        p_min, p_maj = self._flow_pca.transform(h_min), self._flow_pca.transform(h_maj)
        self._h_scaler = StandardScaler().fit(np.concatenate([p_min, p_maj]))
        # torch copies of the transform so that Stage 3 can back-propagate through it
        f32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=DEVICE)
        self._pca_mean_t, self._pca_comp_t = f32(self._flow_pca.mean_), f32(self._flow_pca.components_)
        self._pca_sd_t = f32(np.sqrt(self._flow_pca.explained_variance_))
        self._sc_mean_t, self._sc_scale_t = f32(self._h_scaler.mean_), f32(self._h_scaler.scale_)

        self.flow_min = StableNVP(pca_k, n_layers=hp.flow_layers, hidden=hp.flow_hidden)
        self.flow_maj = StableNVP(pca_k, n_layers=hp.flow_layers, hidden=hp.flow_hidden)
        _train_flow(self.flow_min, f32(self._h_scaler.transform(p_min)).cpu(), hp.flow_epochs, hp.flow_lr, DEVICE)
        _train_flow(self.flow_maj, f32(self._h_scaler.transform(p_maj)).cpu(), hp.flow_epochs, hp.flow_lr, DEVICE)
        _cuda_clear()

    def _scale_h_t(self, h):
        z = (h - self._pca_mean_t) @ self._pca_comp_t.T / self._pca_sd_t
        return (z - self._sc_mean_t) / self._sc_scale_t

    def _flow_score_t(self, h):
        s = self._scale_h_t(h)
        return self.flow_min.log_prob(s) - self.flow_maj.log_prob(s)

    def _flow_score(self, h, batch=8192):
        out = []
        with torch.no_grad():
            for s in range(0, len(h), batch):
                out.append(self._flow_score_t(h[s:s + batch]).cpu())
        return torch.cat(out).numpy()

    def _refine(self, z, steps, lr, batch=4096):
        """Stage 3: maximise the flow likelihood ratio w.r.t. z with G frozen."""
        out = []
        cmin = None
        for s in range(0, len(z), batch):
            zb = z[s:s + batch].clone().to(DEVICE).requires_grad_(True)
            cmin = self._cond(len(zb), 'minority')
            opt = torch.optim.Adam([zb], lr=lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=lr * 0.05)
            for _ in range(steps):
                loss = -self._flow_score_t(self._G.shared(torch.cat([zb, cmin], 1))).mean()
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                with torch.no_grad():
                    zb.clamp_(-3.5, 3.5)
            out.append(zb.detach().cpu())
        return torch.cat(out)

    def sample_rare(self, n, df_real=None, hp: Hyperparams = None, use_flow=True, use_refine=True,
                    use_judge=True, use_maxmin=True, return_diagnostics=False, judge=None):
        """Stages 1-5 of the rare-minority sampler (Stage 6, quantile matching, is applied
        by the caller). The use_* flags switch stages off for the ablation study.
        judge: the Stage-4 judge to use. If None and use_judge, the model's own judge is used
        (dp mode), or a non-private random forest is trained on df_real (legacy mode)."""
        hp = hp or self.hp
        assert (self.flow_min is not None) or not use_flow, "call build_flows() first"
        rng = np.random.RandomState(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        clf, feat_cols = None, self._feat_cols
        if use_judge:
            clf = judge if judge is not None else self.judge
            if clf is None and df_real is not None and df_real[self._target_col].nunique() >= 2:
                clf = RFJudge(self._target_col, feat_cols).fit(df_real)

        pool_sz = max(hp.rare_pool_size, 2 * n)
        all_raw, all_h, all_score = [], [], []
        diag = {"pool_score": [], "refined_before": [], "refined_after": [], "judge_prob": []}
        for _ in range(hp.rare_max_iters):
            if sum(len(r) for r in all_raw) >= 2 * n:
                break
            # Stage 1 — candidate pool
            z = torch.randn(pool_sz, self.latent_dim)
            h = self._hidden(z, 'minority')
            # Stage 2 — flow likelihood-ratio score
            score = self._flow_score(h) if use_flow else rng.rand(pool_sz)
            if use_flow and len(diag["pool_score"]) < 50000:
                diag["pool_score"].append(score[:20000])
            # Stage 3 — latent refinement of the top-scoring fraction
            if use_flow and use_refine and hp.refine_steps > 0 and hp.refine_top_frac > 0:
                k_ref = max(1, int(hp.refine_top_frac * pool_sz))
                idx_ref = np.argsort(-score)[:k_ref]
                z[idx_ref] = self._refine(z[idx_ref], hp.refine_steps, hp.refine_lr)
                h_ref = self._hidden(z[idx_ref], 'minority')
                new_s = self._flow_score(h_ref)
                diag["refined_before"].append(score[idx_ref][:20000].copy())
                diag["refined_after"].append(new_s[:20000])
                h[torch.as_tensor(idx_ref, device=h.device)] = h_ref
                score[idx_ref] = new_s
            # keep the better half by score (or a random half without flows)
            k_keep = min(pool_sz, max(2 * n, pool_sz // 2))
            top_idx = np.argsort(-score)[:k_keep]
            with torch.no_grad():
                raw = self._G.out(h[torch.as_tensor(top_idx, device=h.device)]).cpu().numpy()
            h_sc = self._scale_h_t(h[torch.as_tensor(top_idx, device=h.device)]).cpu().numpy() if use_flow \
                else h[torch.as_tensor(top_idx, device=h.device)].cpu().numpy()
            # Stage 4 — RF judge
            mask = np.ones(len(raw), dtype=bool)
            if clf is not None:
                probs = clf.score(raw, self._decode(raw))
                thr = float(np.percentile(probs, 100 * (1 - hp.judge_keep_frac)))
                if clf.has_prob:
                    thr = max(thr, hp.judge_min_prob)
                mask = probs >= thr
                diag["judge_prob"].append(probs[:20000])
            if mask.sum() == 0:
                continue
            all_raw.append(raw[mask]); all_h.append(h_sc[mask]); all_score.append(score[top_idx][mask])

        if not all_raw:
            df_fb = self.sample_minority_plain(n)
            return (df_fb, {}) if return_diagnostics else df_fb

        raw_all, h_all, s_all = np.concatenate(all_raw), np.concatenate(all_h), np.concatenate(all_score)
        k_sel = min(n, len(raw_all))
        # Stage 5 — diversity selection
        if use_maxmin:
            sel = greedy_maxmin_pca(h_all, k=k_sel, scores=s_all)
        elif use_flow:
            sel = np.argsort(-s_all)[:k_sel]
        else:
            sel = rng.choice(len(raw_all), k_sel, replace=False)
        df_sel = self._decode(raw_all[sel])
        df_sel[self._target_col] = 1
        diag["selected_score"] = [s_all[sel]]
        diag = {k: (np.concatenate(v) if v else np.array([])) for k, v in diag.items()}
        return (df_sel, diag) if return_diagnostics else df_sel


#  SECTION 9 ─ OPTUNA HYPERPARAMETER TUNING

def target_cat_cols(df, feat_cols, target, max_card=20):
    return [c for c in feat_cols if c != target and df[c].nunique() <= max_card]


def optuna_tune(epsilon, gan_train_df, val_df, mia_holdout_df, sc_val, X_val, y_val,
                target, feat_cols, cat_cols, base_hp: Hyperparams, n_trials=8, seed=42):
    """Tunes gan_epochs / learning rates / batch size / (max_grad_norm if DP) against
    validation TSTR AUC penalised by KS/JS distance and excess kNN-MIA AUC."""
    if optuna is None:
        print("  [Optuna] not installed — skipping tuning, using default hyperparams.")
        return base_hp

    def objective(trial):
        hp = Hyperparams(**asdict(base_hp))
        hp.gan_epochs = trial.suggest_int("gan_epochs", 20, max(30, base_hp.gan_epochs), step=10)
        hp.gen_lr = trial.suggest_float("gen_lr", 5e-5, 5e-4, log=True)
        hp.disc_lr = trial.suggest_float("disc_lr", 5e-5, 5e-4, log=True)
        hp.batch_size = trial.suggest_categorical("batch_size", [256, 512])
        if epsilon is not None:
            hp.dp_max_grad_norm = trial.suggest_float("dp_max_grad_norm", 1.0, 10.0)
        else:
            hp.r1_gamma = trial.suggest_float("r1_gamma", 5.0, 15.0)
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

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"  [Optuna] best score={study.best_value:.4f}  params={best}")
    out_hp = Hyperparams(**asdict(base_hp))
    for k, v in best.items():
        setattr(out_hp, k, v)
    return out_hp


#  SECTION 10 ─ SINGLE-RUN PIPELINE (quick use; the paper grid is experiments/run_experiments.py)

def run_pipeline(cfg: DatasetConfig, epsilon: Optional[float], hp: Hyperparams = None,
                 tune: bool = False, n_optuna_trials: int = 8, out_dir: str = "output"):
    hp = hp or Hyperparams()
    os.makedirs(out_dir, exist_ok=True)

    df, target, cat_cols = load_dataset(cfg)
    feat = [c for c in df.columns if c != target]
    cont_cols = [c for c in feat if c not in cat_cols]
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
    model.fit(aug_df, cat_cols, target, ds_name=cfg.name)

    n_aug = len(aug_df)
    common_df = model.sample_common(n_aug)
    common_df = postprocess_synthetic(common_df, aug_df, target, cat_cols, cfg.int_cols, cfg.hard_bounds)
    if hp.privacy_filter_enabled:
        common_df = privacy_realism_filter(common_df, aug_df, feat, desired_n=n_aug, label="common",
                                           max_ref=hp.privacy_max_ref)

    rare_df = pd.DataFrame()
    try:
        model.build_flows(hp)
        n_real_min = model.stats_info["n_hat"] if hp.privacy_mode == "dp" else int(aug_df[target].sum())
        n_rare_need = max(200, int(n_real_min * hp.rare_multiplier))
        rare_df = model.sample_rare(n_rare_need, df_real=aug_df, hp=hp)
        if hp.quantile_blend > 0:
            rare_df = quantile_match(rare_df, real_minority, cont_cols, blend=hp.quantile_blend, orig_df=aug_df)
        rare_df[target] = 1
        rare_df = postprocess_synthetic(rare_df, aug_df, target, cat_cols, cfg.int_cols, cfg.hard_bounds)
        if hp.privacy_filter_enabled:
            rare_df = privacy_realism_filter(rare_df, real_minority, feat, desired_n=n_rare_need,
                                             label="rare", max_ref=hp.privacy_max_ref)
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
        "epsilon": epsilon, "actual_epsilon": model.epsilon_total, "dp_info": model.dp_info,
        "privacy_mode": hp.privacy_mode, "budget": model.budget,
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
    out_csv = os.path.join(out_dir, f"synthetic_{cfg.name}_eps{epsilon}.csv")
    combined_df.to_csv(out_csv, index=False)
    with open(os.path.join(out_dir, f"metrics_{cfg.name}_eps{epsilon}.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n  Saved synthetic data -> {out_csv}")
    return metrics, combined_df


#  SECTION 11 ─ CLI

def main():
    p = argparse.ArgumentParser(description="DP-CTGAN single run — works on any tabular CSV.")
    p.add_argument("--data", type=str, default=None, help="Path to a CSV file (overrides --preset).")
    p.add_argument("--target", type=str, default=None)
    p.add_argument("--preset", type=str, default=None, choices=list(DATASET_PRESETS.keys()))
    p.add_argument("--epsilon", type=str, default="none", help="'none' for no-DP, or a number like 3 / 10.")
    p.add_argument("--delta", type=float, default=None, help="Default: min(1e-5, 1/(10 n_train)).")
    p.add_argument("--tune", action="store_true")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--gan-epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["legacy", "dp"], default="legacy",
                   help="'dp': every component that reads real data is covered by epsilon (see apply_privacy_mode)")
    p.add_argument("--out", type=str, default="output")
    args = p.parse_args()

    set_seed(args.seed)
    if args.data:
        cfg = DatasetConfig(name=os.path.splitext(os.path.basename(args.data))[0],
                            path=args.data, target_col=args.target)
        hp = Hyperparams(**asdict(HYPERPARAMS))
    elif args.preset:
        cfg = DATASET_PRESETS[args.preset]
        hp = hyperparams_for(args.preset)
    else:
        cfg = DATASET_CONFIG
        hp = hyperparams_for(cfg.name)
        print(f"  No --data or --preset given, using preset '{cfg.name}'.")

    epsilon = None if args.epsilon.lower() in ("none", "nodp", "no-dp") else float(args.epsilon)
    if args.gan_epochs:
        hp.gan_epochs = args.gan_epochs
    if args.delta is not None:
        hp.delta = args.delta
    apply_privacy_mode(hp, args.mode)
    run_pipeline(cfg, epsilon, hp=hp, tune=args.tune, n_optuna_trials=args.trials, out_dir=args.out)


if __name__ == "__main__":
    main()
