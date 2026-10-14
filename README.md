# DP-CTGAN — Differentially-Private Tabular Synthesis with a Rare-Minority Sampler

One pipeline, any dataset. `dp_ctgan_pipeline.py` trains a differentially-private conditional
CTGAN (Opacus DP-SGD) with a minority-class-aware **rare sampler** (normalizing-flow scoring,
latent refinement, classifier judge, MaxMin diversity selection, quantile matching), a
nearest-neighbour **PrivacyGate**, and a full evaluation suite: TSTR utility (RF and k-NN),
fidelity (KS, JS, correlation difference, discriminator AUC) and privacy (kNN-MIA, DOMIAS-style
density-ratio attack, LR-MIA, DCR share).

The `experiments/` folder reproduces every table and figure of the TIST manuscript on four
datasets — Adult Census, Taiwan Credit Default, Credit Card Fraud and MIMIC-III — over
No-DP, ε = 1, 3, 10 and five seeds.

| Document | Use it for |
|---|---|
| `DP-CTGAN_Experiment_Guide.docx` | Step-by-step commands to run everything end to end on the Google Cloud VM, and which output goes where in the manuscript |
| `RUNNING_EXPERIMENTS.md` | Background: what changed in the code, the evaluation protocol, VM creation, troubleshooting |
| this README | Overview, repository layout, quick reference |

---

## Repository layout

```
dp_ctgan_pipeline.py           model, rare sampler, PrivacyGate, metrics, dataset presets (DATASET_PRESETS, HP_PRESETS)
requirements.txt               pinned versions (Python 3.11)
data/
  prepare_public_datasets.py   downloads Adult, Taiwan (UCI) and Credit Card Fraud (Kaggle, OpenML fallback)
  mimic3_cohort.sql            MIMIC-III cohort query (BigQuery Standard SQL)
  prepare_mimic3.py            runs the query and writes data/mimic3.csv
experiments/
  common.py                    shared splits (64/8/8/20) and evaluation
  tune.py                      Optuna tuning per dataset and ε -> results/hparams/*.json
  run_experiments.py           the resumable grid: datasets × ε × seeds (+ ablations)
  run_baselines.py             synthcity / SmartNoise / SMOTE baselines on identical splits
  aggregate.py                 LaTeX tables, CSVs, Wilcoxon tests -> results/summary/
  make_figures.py              all figures (PDF + PNG) -> results/figures/
gcp/
  create_vm.sh, setup_vm.sh    create and provision the GPU VM
  run_grid.sh                  launch grid workers in tmux
  sync_results.sh              back results up to a private Cloud Storage bucket
```

---

## Reproducing the paper (summary)

Run on the GPU VM, from the repository root, with the Python environment activated.
The guide (`DP-CTGAN_Experiment_Guide.docx`) explains every step and the expected output.

```bash
# 1. Data
python3 data/prepare_public_datasets.py                    # adult.csv, UCI_Credit_Card.csv, creditcard.csv
gcloud auth application-default login --no-launch-browser  # PhysioNet-linked Google account
python3 data/prepare_mimic3.py --project YOUR_PROJECT_ID    # mimic3.csv

# 2. Smoke test and timing run
python3 experiments/run_experiments.py --datasets adult mimic3 --eps 3 --seeds 0 --quick --ablations --out results_smoke
python3 experiments/run_experiments.py --eps 3 --seeds 0 --ablations --out results_timing

# 3. Tuning (4 datasets in parallel)
mkdir -p logs
for ds in adult taiwan_credit credit_fraud mimic3; do
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=4 \
    python3 experiments/tune.py --datasets $ds --trials 20 > logs/tune_$ds.log 2>&1 &
done; wait

# 4. Full grid: 4 datasets × {No-DP, 1, 3, 10} × 5 seeds = 80 runs (3 workers)
for i in 0 1 2; do
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=5 LOKY_MAX_CPU_COUNT=5 \
    python3 experiments/run_experiments.py --shard $i/3 --ablations > logs/worker_$i.log 2>&1 &
done; wait

# 5. Baselines on the same splits: 3 seeds; CTGAN / TVAE / DP-GAN trained for 300 epochs (--n-iter)
python3 experiments/run_baselines.py --methods pategan dpgan aim --eps 1 3 10 --seeds 0 1 2
python3 experiments/run_baselines.py --methods ctgan tvae --eps none --seeds 0 1 2
python3 experiments/run_baselines.py --methods smote --eps none
python3 experiments/check_outputs.py          # verifies datasets, tuning, grid, artifacts, baselines, tables

# 6. Tables and figures
python3 experiments/aggregate.py --results results --ablation-eps 3
python3 experiments/make_figures.py --results results --fig-eps 3
```

Every run writes its own `metrics.json` and finished runs are skipped, so any step can be
restarted after a Spot preemption or a crash by rerunning the same command.

### Outputs

| File | Manuscript |
|---|---|
| `results/summary/tab_utility.tex` | Table *utility* (TSTR, mean ± std over 5 seeds) |
| `results/summary/tab_fidelity_privacy.tex` | Table *fidelity* (KS, JS, corr., disc. AUC, MIAs, DCR) |
| `results/summary/tab_ablation.tex` | Table *ablation* |
| `results/summary/tab_baselines.tex` | Baseline comparison |
| `results/summary/tab_gate.tex` | PrivacyGate keep / near-copy / outlier rates |
| `results/summary/tab_dp.tex` | δ, σ, q, T and achieved ε |
| `results/summary/tab_confusion.tex` | Appendix confusion matrices |
| `results/summary/stats.txt` | Wilcoxon tests, combined vs common |
| `results/summary/runs.csv`, `mean_std.csv` | All numbers, per run and aggregated |
| `results/figures/fig_*.pdf` | All figures (privacy–utility, ours vs baselines, gains, ablation, MIA ROC, sampler scores, PR curves, marginals, t-SNE, correlation differences); drawn at print size (6.8 in = `\textwidth`), error bands = 95% CI over seeds |

### Evaluation protocol (per seed, stratified)

- Splits: 20% sealed test; of the remaining 80%: 10% membership holdout, 10% validation,
  80% GAN training (64% of all records).
- F1 threshold chosen on the validation split. RF (200 trees, balanced) and k-NN (k = 5).
- δ = min(10⁻⁵, 1/(10·n_train)). Hyperparameter tuning on the private data is **not**
  included in the reported ε.

---

## Using the pipeline on your own data

```bash
pip install -r requirements.txt

# Point it at any CSV — categorical columns are auto-detected
python3 dp_ctgan_pipeline.py --data yourfile.csv --target your_label_column --epsilon 3

# Or use a built-in preset
python3 dp_ctgan_pipeline.py --preset adult --epsilon 3
python3 dp_ctgan_pipeline.py --preset taiwan_credit --epsilon none   # no-DP baseline
python3 dp_ctgan_pipeline.py --preset mimic3 --epsilon 10 --tune --trials 15
```

Outputs land in `output/`: the synthetic dataset (CSV) and a metrics JSON.

For full control, add a `DatasetConfig` to `DATASET_PRESETS` in `dp_ctgan_pipeline.py`:

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

GAN, flow and training hyperparameters live in the `Hyperparams` dataclass; per-dataset
defaults are in `HP_PRESETS`.

| Stays the same across datasets | You configure per dataset |
|---|---|
| DPCTGAN engine (GAN + Opacus DP-SGD) | file path, target column |
| StableNVP rare-class flow sampler | categorical / log1p / int columns |
| PrivacyGate near-copy/outlier filter | numeric hard bounds |
| kNN-MIA / DOMIAS / LR-MIA / DCR evaluation | ε budget, GAN hyperparameters |
| TSTR RF/k-NN evaluation | custom cleaning hook (optional) |

---

## Data governance (MIMIC-III)

MIMIC-III is governed by the PhysioNet Credentialed Health Data Use Agreement. Keep
`data/mimic3.csv` and `results/mimic3/**/synthetic_*.csv.gz` only on the VM, your own machine or
a **private** bucket. Never commit them (`.gitignore` excludes `data/*.csv` and `results/`), and
do not share or upload synthetic MIMIC-III data. Publishing aggregate statistics is allowed.

## License

MIT — see LICENSE.
