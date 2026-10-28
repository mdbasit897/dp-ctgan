# RareSel-DP — Differentially Private Tabular Synthesis with Private Rare-Minority Selection

Code and results for the paper *RareSel-DP: Private Selection of Rare Minority Records for
Differentially Private Tabular Data Synthesis*.

`dp_ctgan_pipeline.py` trains a conditional tabular GAN whose discriminator is trained with DP-SGD and
adds a **rare-minority stream**: minority-conditioned candidates drawn from the trained generator,
scored by a private minority classifier (the *judge*) and thinned by greedy max–min diversity
selection. The evaluation suite covers TSTR utility (random forest and k-NN), fidelity (KS, JS,
correlation difference) and empirical privacy (kNN-MIA, DOMIAS, LR-MIA, DCR share).

## Privacy

RareSel-DP (`--mode dp`) releases data that are (ε, δ)-DP end to end. One budget is split between the
four mechanisms that read training records, each at δ/4:

| Mechanism | Share of ε | What it releases |
|---|---|---|
| Column encoder | 0.05 | Gaussian-mechanism histograms of the continuous columns: private ranges for columns without a public bound (30% of the share) and 64-bin histograms on every column's range (70%); the GMM mode encoder is fitted to these histograms |
| Discriminator | 0.75 | DP-SGD (Opacus, RDP accountant); the generated half of every batch has a fixed size, so each step has sensitivity C |
| Minority statistics | 0.10 | noisy minority count, mean and standard deviation (Gaussian mechanism) |
| Judge | 0.10 | DP-SGD MLP minority classifier |

The schema is public and never computed from the records: the category domain of every categorical
column (`CAT_DOMAINS`, from the dataset documentation and the published MIMIC-III cohort query) and
documented bounds of continuous columns (`PUBLIC_BOUNDS`). The training-set size N is treated as
public. The generator, both synthetic streams, the sampler and all clipping (to the released ranges)
are post-processing. Hyperparameters are either the same defaults for every dataset (`Hyperparams`)
or chosen by leave-one-dataset-out transfer (`experiments/tune_lodo.py`): the configuration used for a
dataset is selected on the other three, so model selection reads none of its records.

`--encoder data` reproduces the public-encoder assumption (encoder and ranges fitted on the training
split outside ε, budget 0.8/0.1/0.1 at δ/3) as an ablation; `--protocol legacy` is the earlier
protocol with per-dataset presets. `--mode nonprivate` is the non-private variant (feature and
correlation matching on real batches, exact minority statistics, a random-forest judge on real data,
quantile matching, PrivacyGate filtering); its output is **not** differentially private.

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
  tune_lodo.py                 leave-one-dataset-out hyperparameter selection -> results_e2e/hparams_lodo/
  tune.py                      Optuna search per dataset and ε (earlier protocol) -> results/hparams/*.json
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

```
results_e2e/                   RareSel-DP, end-to-end protocol, LODO hyperparameters: 4 datasets x {No-DP, 1, 3, 10} x 5 seeds
results_e2e/hparams_lodo/      candidate scores, selected configurations (one JSON per dataset and setting)
results_e2e_untuned/           the same grid with the global defaults (no tuning at all)
results_e2e_pubenc/            ablation: same hyperparameters, encoder fitted on the training split (outside eps)
results_e2e/baselines/         SmartNoise DP-CTGAN / PATE-CTGAN / MST (end-to-end, *_e2e), CTGAN, TVAE, SMOTE
```

Each run folder contains `metrics.json` (utility, fidelity, attacks, PrivacyGate audit, the DP
accounting of every component including the encoder, the released column ranges and their source,
sampler diagnostics, timings). Synthetic data, per-record predictions and attack scores are not
published; the MIMIC-III-derived ones may not be shared under the PhysioNet data use agreement.

## Reproducing the experiments

Run on a GPU machine from the repository root (`bash gcp/setup_vm.sh` prepares a Google Cloud VM).

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 1. Data
python3 data/prepare_public_datasets.py                    # adult.csv, UCI_Credit_Card.csv, creditcard.csv
python3 data/prepare_mimic3.py --project YOUR_PROJECT_ID    # mimic3.csv (PhysioNet-linked Google account)

# 2. Leave-one-dataset-out hyperparameter selection (24 candidates x 4 datasets x 4 settings)
for i in 0 1 2 3 4 5 6 7; do
  python3 experiments/tune_lodo.py score --shard $i/8 > logs/lodo_$i.log 2>&1 &
done; wait
python3 experiments/tune_lodo.py select

# 3. RareSel-DP grids
for i in 0 1 2 3 4 5 6 7; do      # main grid with ablations
  python3 experiments/run_experiments.py --shard $i/8 --ablations --hp-dir results_e2e/hparams_lodo \
    --out results_e2e > logs/e2e_$i.log 2>&1 &
done; wait
for i in 0 1 2 3 4 5 6 7; do      # no tuning, and the public-encoder ablation
  python3 experiments/run_experiments.py --shard $i/8 --hp-dir none --out results_e2e_untuned > logs/untuned_$i.log 2>&1 &
done; wait
for i in 0 1 2 3 4 5 6 7; do
  python3 experiments/run_experiments.py --shard $i/8 --encoder data --hp-dir results_e2e/hparams_lodo \
    --out results_e2e_pubenc > logs/pubenc_$i.log 2>&1 &
done; wait

# 4. Baselines on the same splits (SmartNoise needs its own environment: smartnoise-synth 1.0.8 pins an older Opacus)
uv venv --python 3.11 .venv-sn
uv pip install --python .venv-sn/bin/python --torch-backend auto smartnoise-synth==1.0.8 scikit-learn pandas scipy
.venv-sn/bin/python experiments/run_baselines.py --methods sn_dpctgan sn_patectgan sn_mst --e2e \
    --seeds 0 1 2 3 4 --out results_e2e/baselines
python3 experiments/run_baselines.py --methods ctgan tvae --eps none --seeds 0 1 2 --out results_e2e/baselines
python3 experiments/run_baselines.py --methods smote --eps none --out results_e2e/baselines

# 5. Tables, figures, checks
python3 experiments/aggregate.py --results results_e2e --baselines-dir results_e2e/baselines --ablation-eps 3
python3 experiments/aggregate.py --results results_e2e_untuned --ablation-eps 3
python3 experiments/aggregate.py --results results_e2e_pubenc --ablation-eps 3
python3 experiments/make_figures.py --results results_e2e --fig-eps 3
python3 experiments/check_outputs.py --results results_e2e --baselines-dir results_e2e/baselines \
    --hp-dir results_e2e/hparams_lodo --baseline-seeds 3
```

Every run writes its own `metrics.json` and finished runs are skipped, so any step can be restarted
after a preemption by running the same command again. A failed run leaves `error.txt` in its folder.

### Evaluation protocol (per seed, stratified)

- Splits: 20% sealed test set; of the remaining 80%, 10% membership-inference holdout, 10% validation,
  and the rest (64% of all records) for training. LODO selection uses only the seed-0 training and
  validation splits of the other datasets.
- TSTR with a random forest (200 trees, balanced class weights) and k-NN (k = 5); F1 at the threshold
  that maximizes F1 on the real validation split (an oracle calibration).
- δ = min(10⁻⁵, 1/(10·n_train)), split δ/4 per private component.

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
    cat_domains={"region": ["north", "south"], "plan_type": ["basic", "plus"]},   # public category domains
    public_bounds={"age": (18, 100)},         # documented bounds; other numeric columns get a private range
    log1p_cols=["income"],                   # skewed numeric columns
    int_cols=["age"],                        # rounded after synthesis
    hard_bounds={"age": (18, 100)},          # physical/logical bounds
)
```

GAN, flow and training hyperparameters live in the `Hyperparams` dataclass (the defaults are the
end-to-end protocol's global setting); `HP_PRESETS` holds the per-dataset presets of the earlier
protocol. Without `cat_domains`, use `--encoder data` (the encoder is then fitted on your data and the
release is not end-to-end private).

## Data governance (MIMIC-III)

MIMIC-III is governed by the PhysioNet Credentialed Health Data Use Agreement. Keep
`data/mimic3.csv` and every MIMIC-III-derived synthetic file on your own machine or in a **private**
bucket. `.gitignore` excludes the data, synthetic data and per-record scores; only aggregate metrics
are published.

## License

MIT — see LICENSE.
