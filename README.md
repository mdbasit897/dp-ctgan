# RareSel-DP — Differentially Private Tabular Synthesis with Private Rare-Minority Selection

Code and results for the paper *RareSel-DP: Private Selection of Rare Minority Records for
Differentially Private Tabular Data Synthesis*.

`dp_ctgan_pipeline.py` trains a conditional tabular GAN whose discriminator is trained with DP-SGD and
adds a **rare-minority stream**: minority-conditioned candidates drawn from the trained generator,
scored by a private minority classifier (the *judge*) and thinned by greedy max–min diversity
selection. The evaluation suite covers TSTR utility (random forest and k-NN), fidelity (KS, JS,
correlation difference) and empirical privacy (kNN-MIA, DOMIAS, LR-MIA, DCR share).

## Privacy

RareSel-DP (`--mode dp`, the default) spends one (ε, δ) budget on the three components that read
training records: the DP-SGD discriminator (0.8 ε), the minority count, mean and standard deviation
released with the Gaussian mechanism (0.1 ε), and a DP-SGD MLP judge (0.1 ε), each at δ/3. In every
discriminator step the generated half of the batch has a fixed size and conditions drawn at the
released minority rate, independent of which real records were sampled, so each step has
sensitivity C. The generator, both synthetic streams and the sampler are post-processing.

The guarantee assumes that the column encoder, the column ranges and the training-set size are
public. In the experiments they are fitted to the training split, and the hyperparameter search is
not included in ε; see the paper's privacy section.

`--mode nonprivate` is the non-private variant reported in the paper as an upper reference and used
as the tuning configuration: it adds feature and correlation matching on real batches, exact
minority statistics, a random-forest judge trained on real data, quantile matching and PrivacyGate
filtering. Its output is **not** differentially private.

## Repository layout

```
dp_ctgan_pipeline.py           model, rare sampler, privacy modes, PrivacyGate audit, metrics, dataset presets
requirements.txt               versions used for the paper (Python 3.12)
data/
  prepare_public_datasets.py   downloads Adult, Taiwan (UCI) and Credit Card Fraud (Kaggle, OpenML fallback)
  mimic3_cohort.sql            MIMIC-III cohort query (BigQuery Standard SQL)
  prepare_mimic3.py            runs the query and writes data/mimic3.csv
experiments/
  common.py                    shared splits (64/8/8/20) and evaluation
  tune.py                      Optuna search per dataset and ε -> results/hparams/*.json
  run_experiments.py           resumable grid: datasets × ε × seeds (+ ablations)
  run_baselines.py             SmartNoise / synthcity / SMOTE baselines on identical splits
  aggregate.py                 CSV summaries, LaTeX tables, Wilcoxon tests -> <results>/summary/
  make_figures.py              all figures (PDF + PNG) -> <results>/figures/
  make_paper_tables.py         the paper's tables from the outputs below
  check_outputs.py             completeness and consistency check of inputs and outputs
gcp/
  setup_vm.sh                  provision a GPU VM (NVIDIA driver, Python environment, repository)
  run_grid.sh                  launch grid workers in tmux
  sync_results.sh              back results up to a private Cloud Storage bucket
results_dp/                    RareSel-DP, tuned hyperparameters: 4 datasets × {No-DP, 1, 3, 10} × 5 seeds
results_dp_untuned/            RareSel-DP with the dataset presets (no tuning)
results/                       non-private variant (<dataset>/), baselines (baselines/),
                               selected hyperparameters (hparams/), AIM attempt (aim_attempt/)
```

## Published outputs

Each run folder contains `metrics.json` (utility, fidelity, attacks, PrivacyGate audit, DP accounting
of every component, sampler diagnostics, timings). `results_dp/summary/` and
`results_dp_untuned/summary/` contain `runs.csv`, `mean_std.csv`, `stats.txt` (all significance
tests) and the LaTeX tables; `results_dp/figures/` contains the figures. Synthetic data, per-record
predictions and attack scores are not published; the MIMIC-III-derived ones may not be shared under
the PhysioNet data use agreement. Every published number can be recomputed from `metrics.json`:

```bash
python3 experiments/aggregate.py --results results_dp --baselines-dir results/baselines --nonprivate-results results --ablation-eps 3
python3 experiments/aggregate.py --results results_dp_untuned --ablation-eps 3
python3 experiments/make_paper_tables.py --out paper_tables
```

## Reproducing the experiments

Run on a GPU machine from the repository root (`bash gcp/setup_vm.sh` prepares a Google Cloud VM).

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 1. Data
python3 data/prepare_public_datasets.py                    # adult.csv, UCI_Credit_Card.csv, creditcard.csv
gcloud auth application-default login --no-launch-browser  # Google account linked to PhysioNet
python3 data/prepare_mimic3.py --project YOUR_PROJECT_ID    # mimic3.csv

# 2. Hyperparameter search (20 Optuna trials per dataset and setting on the seed-0 split,
#    non-private tuning configuration; see the paper's appendix)
mkdir -p logs
for ds in adult taiwan_credit credit_fraud mimic3; do
  python3 experiments/tune.py --datasets $ds --trials 20 > logs/tune_$ds.log 2>&1 &
done; wait

# 3. RareSel-DP grid with ablations (80 runs; 4 workers)
for i in 0 1 2 3; do
  python3 experiments/run_experiments.py --mode dp --shard $i/4 --ablations \
    --hp-dir results/hparams --out results_dp > logs/dp_$i.log 2>&1 &
done; wait
#    the same grid with the dataset presets and no tuning
for i in 0 1 2 3; do
  python3 experiments/run_experiments.py --mode dp --shard $i/4 --hp-dir none \
    --out results_dp_untuned > logs/untuned_$i.log 2>&1 &
done; wait
#    non-private variant (upper reference)
python3 experiments/run_experiments.py --mode nonprivate --ablations --hp-dir results/hparams --out results

# 4. Baselines on the same splits
python3 experiments/run_baselines.py --methods ctgan tvae --eps none --seeds 0 1 2
python3 experiments/run_baselines.py --methods smote --eps none
#    AIM (synthcity), reported in the appendix only: 3-4 h per run on Adult
python3 experiments/run_baselines.py --methods aim --datasets adult --eps 1 3 10 --seeds 0 --out results/aim_attempt
python3 experiments/run_baselines.py --methods aim --datasets adult --eps 1 --seeds 1 2 --out results/aim_attempt
#    SmartNoise needs its own environment (smartnoise-synth 1.0.8 pins an older Opacus)
uv venv --python 3.11 .venv-sn
uv pip install --python .venv-sn/bin/python --torch-backend auto smartnoise-synth==1.0.8 scikit-learn pandas scipy
.venv-sn/bin/python experiments/run_baselines.py --methods sn_dpctgan sn_patectgan sn_mst --eps 1 3 10 --seeds 0 1 2
.venv-sn/bin/python experiments/run_baselines.py --methods sn_dpctgan sn_patectgan sn_mst --eps 1 3 10 --matched --seeds 0 1 2 3 4

# 5. Tables, figures, checks
python3 experiments/aggregate.py --results results_dp --baselines-dir results/baselines --nonprivate-results results --ablation-eps 3
python3 experiments/aggregate.py --results results_dp_untuned --ablation-eps 3
python3 experiments/make_figures.py --results results_dp --fig-eps 3
python3 experiments/check_outputs.py --results results_dp --baselines-dir results/baselines
python3 experiments/make_paper_tables.py --out paper_tables
```

Every run writes its own `metrics.json` and finished runs are skipped, so any step can be restarted
after a preemption by running the same command again. A failed run leaves `error.txt` in its folder.

### Evaluation protocol (per seed, stratified)

- Splits: 20% sealed test set; of the remaining 80%, 10% membership-inference holdout, 10% validation,
  and the rest (64% of all records) for training.
- TSTR with a random forest (200 trees, balanced class weights) and k-NN (k = 5); F1 at the threshold
  that maximizes F1 on the real validation split (an oracle calibration).
- δ = min(10⁻⁵, 1/(10·n_train)), split δ/3 per private component.

## Using the pipeline on your own data

```bash
python3 dp_ctgan_pipeline.py --data yourfile.csv --target your_label_column --epsilon 3
python3 dp_ctgan_pipeline.py --preset adult --epsilon 3
python3 dp_ctgan_pipeline.py --preset taiwan_credit --epsilon none   # no DP
```

Outputs land in `output/`: the synthetic dataset (CSV) and a metrics JSON. For full control, add a
`DatasetConfig` to `DATASET_PRESETS` in `dp_ctgan_pipeline.py`:

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

GAN, flow and training hyperparameters live in the `Hyperparams` dataclass; per-dataset defaults
are in `HP_PRESETS`.

## Data governance (MIMIC-III)

MIMIC-III is governed by the PhysioNet Credentialed Health Data Use Agreement. Keep
`data/mimic3.csv` and every MIMIC-III-derived synthetic file on your own machine or in a **private**
bucket. `.gitignore` excludes the data, synthetic data and per-record scores; only aggregate metrics
are published.

## License

MIT — see LICENSE.
