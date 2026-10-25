# RareSel-DP — Differentially Private Tabular Synthesis with Private Rare-Minority Selection

One pipeline, any dataset. `dp_ctgan_pipeline.py` trains a differentially private conditional
CTGAN and adds a **rare-minority stream** chosen by a five-stage sampler (normalizing-flow
scoring, latent refinement, a private classifier judge, MaxMin diversity selection), with a
full evaluation suite: TSTR utility (RF and k-NN), fidelity (KS, JS, correlation difference,
discriminator AUC) and privacy (kNN-MIA, DOMIAS-style density-ratio attack, LR-MIA, DCR share).

**Privacy (the paper's method, `--mode dp`).** Every component that reads training records is
covered by one (ε, δ) budget: the DP-SGD discriminator (0.8 ε), the minority mean/std released
with the Gaussian mechanism (0.1 ε) and a DP-SGD MLP judge (0.1 ε), each at δ/3; the sampler is
post-processing. The GMM mode encoder, the column ranges used for clipping and the training-set
size are taken from the training data and treated as public, as in the other DP-CTGAN
implementations. Hyperparameter tuning is **not** included in the reported ε.
`--mode legacy` is the first experiment round (non-private feature matching, random-forest judge,
quantile matching); it is kept only to reproduce the round-1 comparison and is not end-to-end DP.

The `experiments/` folder reproduces every table and figure of the manuscript on four
datasets — Adult Census, Taiwan Credit Default, Credit Card Fraud and MIMIC-III — over
No-DP, ε = 1, 3, 10 and five seeds.

| Document | Use it for |
|---|---|
| `ROUND2_DP_RUN.md` | The privatised pipeline (`--mode dp`): what it changes, the main grid, SmartNoise baselines, tables and figures |
| `REVIEW_RERUN.md` | The re-runs made for the internal review: sampler padding fix, matched SmartNoise baselines, untuned grid |
| `RUNNING_EXPERIMENTS.md` | Background: evaluation protocol, VM creation, synthcity baselines, troubleshooting (round 1) |
| `DP-CTGAN_Experiment_Guide.docx` | Round-1 step-by-step guide |
| this README | Overview, repository layout, quick reference |

---

## Repository layout

```
dp_ctgan_pipeline.py           model, rare sampler, privacy modes, PrivacyGate audit, metrics, dataset presets
requirements.txt               pinned versions (Python 3.11)
data/
  prepare_public_datasets.py   downloads Adult, Taiwan (UCI) and Credit Card Fraud (Kaggle, OpenML fallback)
  mimic3_cohort.sql            MIMIC-III cohort query (BigQuery Standard SQL)
  prepare_mimic3.py            runs the query and writes data/mimic3.csv
experiments/
  common.py                    shared splits (64/8/8/20) and evaluation
  tune.py                      Optuna tuning per dataset and ε -> results/hparams/*.json
  run_experiments.py           the resumable grid: datasets × ε × seeds (+ ablations); --mode dp|legacy
  run_baselines.py             SmartNoise / synthcity / SMOTE baselines on identical splits (--matched)
  aggregate.py                 LaTeX tables, CSVs, Wilcoxon tests -> <results>/summary/
  make_figures.py              all figures (PDF + PNG) -> <results>/figures/
  check_outputs.py             completeness and consistency check of all inputs and outputs
gcp/
  setup_vm.sh                  provision the GPU VM
  run_grid.sh                  launch grid workers in tmux
  sync_results.sh              back results up to a private Cloud Storage bucket
```

---

## Reproducing the paper (summary)

Run on the GPU VM, from the repository root, with the Python environment activated.
`ROUND2_DP_RUN.md` and `REVIEW_RERUN.md` explain every step and the expected output.

```bash
# 1. Data
python3 data/prepare_public_datasets.py                    # adult.csv, UCI_Credit_Card.csv, creditcard.csv
gcloud auth application-default login --no-launch-browser  # PhysioNet-linked Google account
python3 data/prepare_mimic3.py --project YOUR_PROJECT_ID    # mimic3.csv

# 2. Tuning (round 1, legacy objective; the dp grid reuses these settings, see the paper's appendix)
mkdir -p logs
for ds in adult taiwan_credit credit_fraud mimic3; do
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=4 \
    python3 experiments/tune.py --datasets $ds --trials 20 > logs/tune_$ds.log 2>&1 &
done; wait

# 3. Main grid, privatised pipeline: 4 datasets × {No-DP, 1, 3, 10} × 5 seeds = 80 runs (4 workers)
for i in 0 1 2 3; do
  nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=5 LOKY_MAX_CPU_COUNT=5 \
    python3 experiments/run_experiments.py --mode dp --shard $i/4 --ablations \
      --hp-dir results/hparams --out results_dp > logs/dp_worker_$i.log 2>&1 &
done; wait
# untuned robustness grid (dataset presets, no ablations)
#   ... run_experiments.py --mode dp --shard $i/4 --hp-dir none --out results_dp_untuned

# 4. Baselines on the same splits (SmartNoise in its own environment, see ROUND2_DP_RUN.md §5.2)
.venv-sn/bin/python experiments/run_baselines.py --methods sn_dpctgan sn_patectgan sn_mst --matched --seeds 0 1 2 3 4
python3 experiments/run_baselines.py --methods smote --eps none

# 5. Tables, figures, checks
python3 experiments/aggregate.py --results results_dp --baselines-dir results/baselines --legacy-results results --ablation-eps 3
python3 experiments/make_figures.py --results results_dp --fig-eps 3
python3 experiments/check_outputs.py --results results_dp --baselines-dir results/baselines
```

Every run writes its own `metrics.json` and finished runs are skipped, so any step can be
restarted after a Spot preemption or a crash by rerunning the same command.

### Outputs (in `results_dp/summary/` and `results_dp/figures/`)

| File | Content |
|---|---|
| `tab_utility.tex` | TSTR utility: TRTR, common stream, combined (mean ± std over 5 seeds) |
| `tab_fidelity_privacy.tex` | KS, JS, correlation difference, MIAs, DCR |
| `tab_ablation.tex` | sampler ablation (every variant adds the same number of minority records) |
| `tab_baselines.tex`, `tab_baseline_minority.tex` | baseline comparison and the baselines' synthetic minority rates |
| `tab_budget.tex`, `tab_dp.tex` | ε per component, basic and RDP totals; DP-SGD accounting of the discriminator |
| `tab_gate.tex`, `tab_confusion.tex`, `tab_dp_vs_legacy.tex` | PrivacyGate audit, confusion matrices, private vs round-1 pipeline |
| `stats.txt` | Wilcoxon tests: all runs, DP runs only, without the tuning seed, seed-level; ablations; padded rare streams |
| `runs.csv`, `mean_std.csv` | all numbers, per run and aggregated |
| `figures/fig_*.pdf` | all figures, drawn at print size (6.8 in = `\textwidth`) |

### Evaluation protocol (per seed, stratified)

- Splits: 20% sealed test; of the remaining 80%: 10% membership holdout, 10% validation,
  80% GAN training (64% of all records).
- F1 threshold chosen on the (real) validation split. RF (200 trees, balanced) and k-NN (k = 5).
- δ = min(10⁻⁵, 1/(10·n_train)), split δ/3 per private component. Hyperparameter tuning on the
  private data is **not** included in the reported ε.

---

## Using the pipeline on your own data

```bash
pip install -r requirements.txt

# Point it at any CSV — categorical columns are auto-detected (privatised pipeline by default;
# add --mode legacy for the round-1 pipeline)
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
| StableNVP rare-class flow sampler + private judge | categorical / log1p / int columns |
| PrivacyGate near-copy/outlier audit | numeric hard bounds |
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
