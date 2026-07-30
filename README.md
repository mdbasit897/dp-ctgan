# DP-CTGAN — Generalized Differentially-Private Tabular Synthesis

One pipeline, any dataset. This merges four dataset-specific research scripts
(Adult Census, Taiwan Credit, Credit Card Fraud, MIMIC-III) into a single
`dp_ctgan_pipeline.py` that trains a differentially-private CTGAN with a
minority-class-aware "rare sampler" (normalizing flows), a nearest-neighbour
PrivacyGate, and kNN-MIA / LR-MIA membership-inference evaluation.

## Quick start

```bash
pip install -r requirements.txt

# Point it at any CSV — target column and categorical columns are auto-detected
python dp_ctgan_pipeline.py --data yourfile.csv --target your_label_column --epsilon 3
```

Outputs land in `output/`: the synthetic dataset (CSV) and a metrics JSON
(TSTR AUC/AUPRC, KS/JS distribution distance, kNN-MIA and LR-MIA privacy
scores).

## Using a built-in preset

```bash
python dp_ctgan_pipeline.py --preset adult --epsilon 3
python dp_ctgan_pipeline.py --preset taiwan_credit --epsilon none   # no-DP baseline
python dp_ctgan_pipeline.py --preset mimic3 --epsilon 10 --tune --trials 15
```

Presets live in `DATASET_PRESETS` near the top of `dp_ctgan_pipeline.py` —
edit the `path` field to point at your local copy of that CSV, or add your
own dataset as a new `DatasetConfig` entry.

## Adapting to your own dataset

You have two options:

1. **Zero config** — just pass `--data` and `--target`; categorical columns,
   ID columns, and skewed numeric columns are inferred automatically. Good
   for a first pass.

2. **Full control** — add a `DatasetConfig` to `DATASET_PRESETS`:

```python
DATASET_PRESETS["my_dataset"] = DatasetConfig(
    name="my_dataset",
    path="my_data.csv",
    target_col="label",
    id_cols=["id", "internal_ref"],          # dropped before training
    cat_cols=["region", "plan_type"],        # None = auto-infer
    log1p_cols=["income"],                   # skewed numeric columns
    int_cols=["age"],                        # rounded after synthesis
    hard_bounds={"age": (18, 100)},          # physical/logical bounds
)
```

then run `python dp_ctgan_pipeline.py --preset my_dataset --epsilon 3`.

## Hyperparameters

All GAN/flow/training hyperparameters live in the `Hyperparams` dataclass
(`gan_epochs`, `latent_dim`, `dp_max_grad_norm`, `quantile_blend`, etc.).
Override defaults via `HYPERPARAMS` at the top of the file, `--gan-epochs`
on the CLI, or `--tune` to have Optuna search a subset of them automatically.

## What's dataset-agnostic vs. what you configure

| Stays the same across datasets | You configure per dataset |
|---|---|
| DPCTGAN engine (GAN + Opacus DP-SGD) | file path, target column |
| StableNVP rare-class flow sampler | categorical / log1p / int columns |
| PrivacyGate near-copy/outlier filter | numeric hard bounds |
| kNN-MIA / LR-MIA evaluation | custom cleaning hook (optional) |
| TSTR RF/KNN evaluation | epsilon budget, GAN hyperparameters |

## License

MIT — see LICENSE.
