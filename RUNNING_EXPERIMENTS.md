# Running the experiments for the TIST manuscript

This guide takes you from an empty Google Cloud project to every table and figure the revised
manuscript needs. Commands are meant to be copied one block at a time. Lines starting with
`#` are comments.

**What you get at the end**

| Output | Fills in (manuscript) |
|---|---|
| `results/summary/tab_utility.tex` | Table *utility* (mean ± std over 5 seeds, validation thresholds) |
| `results/summary/tab_fidelity_privacy.tex` | Table *fidelity* (KS, JS, correlation, discriminator AUC, 3 attacks, DCR) |
| `results/summary/tab_ablation.tex` | Table *ablation* (currently empty) |
| `results/summary/tab_baselines.tex` | Baseline comparison (replaces the PATE-GAN / DP-CGAN tables) |
| `results/summary/tab_gate.tex`, `tab_dp.tex` | PrivacyGate rates; δ, σ, q, T, achieved ε (Table *hyper* TODO row) |
| `results/summary/tab_confusion.tex` | Appendix confusion matrices (validation thresholds) |
| `results/summary/stats.txt` | Wilcoxon tests for "combined vs common" claims |
| `results/figures/*.pdf` | All figures (see section 13) |

---

## 0. Overview of the workflow

```
laptop                                   Google Cloud
──────                                   ────────────
1. push code to a private GitHub repo  →  2. create GPU VM (gcp/create_vm.sh)
                                          3. set up VM      (gcp/setup_vm.sh)
                                          4. prepare data   (data/*.py, BigQuery for MIMIC-III)
                                          5. smoke test → timing run → tuning → full grid → baselines
                                          6. aggregate tables + figures
7. copy results/summary and results/figures back, paste into the manuscript
8. delete the VM
```

---

## 1. What changed in the code, and why

The previous `dp_ctgan_pipeline.py` did not implement several things the manuscript describes,
and it had bugs. Everything below is now in the code, so **re-run all experiments**. Old numbers
will not be reproduced exactly.

| Change | Why it matters for the paper |
|---|---|
| **Stage 3 latent refinement** implemented (`DPCTGAN._refine`) | It was described in the paper and drawn in Figure 3 but was missing from the code |
| **L_corr and L_MFM** losses added; λ values in `Hyperparams` (`lambda_fm=1, lambda_mm=2, lambda_corr=1, lambda_mfm=1`) | Described in the paper but missing. Report the λ values (manuscript TODO) |
| **True class conditioning**: real rows are fed to the discriminator with their own label; previously every row was labelled "majority" | Makes the model a real conditional GAN, as the paper says |
| Target column removed from the encoded features (the label is carried only by the condition) | It used to be modelled as a continuous GMM column |
| **Fine-tuning with early stopping** (`finetune_epochs`, `finetune_patience`) | Matches the paper's description |
| **Per-dataset settings** from the paper's appendix table (`HP_PRESETS`) | Pool sizes, flow width, refinement, judge, blend and PrivacyGate reference size now follow the paper |
| **Flow bug fixed**: coupling layers now alternate which half they transform | Before, half of the hidden coordinates were never modelled by the flow |
| **PrivacyGate**: near-copies are never re-admitted; keep/near-copy/outlier rates are returned | It used to refill with random rejects, including near-copies |
| **Quantile matching** only on continuous columns, bounded by the training data | It used to touch categorical codes and the test-set range |
| Adult: log-transformed `capital-gain/loss` are no longer rounded to integers | Rounding destroyed those columns |
| Taiwan: `BILL_AMT1–6` use **signed** log1p | Negative balances were clipped to 0 |
| Credit Card Fraud: `max_rows=100_000` keeps all 492 frauds and subsamples the majority class | Reproduces the paper's setup (test set of 20,000 with 98 frauds). Set `None` to use all rows |
| **Optuna bug fixed**: the tuned clipping norm `dp_max_grad_norm` is now actually applied | The parameter name did not match before |
| δ now defaults to `min(1e-5, 1/(10·n_train))`; σ, q, steps and achieved ε are logged | Required for the DP table |
| Vectorised encoder/decoder, GPU MaxMin | Much faster; the per-row Python loops dominated runtime |

New scripts:

```
data/prepare_public_datasets.py   Adult, Taiwan (UCI) and Credit Card Fraud (Kaggle or OpenML)
data/mimic3_cohort.sql            MIMIC-III cohort query for BigQuery
data/prepare_mimic3.py            runs that query and writes data/mimic3.csv
experiments/common.py             shared splits + evaluation (validation thresholds, MIAs, DCR, fidelity)
experiments/tune.py               Optuna tuning per dataset and ε (seed 0)
experiments/run_experiments.py    the resumable grid: datasets × ε × seeds (+ ablations)
experiments/run_baselines.py      baselines on identical splits and evaluation
experiments/aggregate.py          LaTeX tables, CSVs, Wilcoxon tests
experiments/make_figures.py       all figures (PDF + PNG)
gcp/*.sh                          create VM, set it up, launch workers, sync results
```

### Evaluation protocol (what to write in the manuscript)

- **Splits (per seed, stratified):** 20% sealed test, then of the remaining 80% pool:
  10% membership holdout, 10% validation, 80% GAN training (**64% of all records**; the
  manuscript currently says 72%, so update Section 5.2).
- **Thresholds:** F1 threshold chosen on the **validation** split, then applied to the test set.
  This resolves the "test-optimal threshold" TODOs.
- **Classifiers:** random forest (200 trees, balanced) and k-NN with **k = 5** (state k = 5 in Section 5.3).
- **Attacks:** kNN-MIA (distance), **DOMIAS-style density ratio** (KDE in a 10-d PCA space;
  reference data = validation split), LR-MIA (confidence), each with AUC and TPR at
  FPR = 0.1%, 1% and 10%, plus a minority-only AUC (a targeted view of the rare records).
  Also reported: **DCR share**, the fraction of synthetic records closer to a training record than
  to a holdout record (0.5 = no memorisation signal).
- **Fidelity:** mean KS, mean JS distance (all records and minority only), mean absolute
  correlation difference, and discriminator (propensity) AUC.
- **Ablations:** the same number of extra minority rows for every variant. Variants: plain
  `G(z, c_min)` samples, without flow + refinement, without refinement, without judge, without
  MaxMin, and without quantile matching.

---

## 2. Time and cost planning (estimate first, then measure)

The grid is 4 datasets × 4 budgets (No-DP, ε = 1, 3, 10) × 5 seeds = **80 runs**, plus baselines.
Runtime per run depends mostly on dataset size and the number of GAN epochs. Do the timing run in
step 8 before you commit to the full grid. As a rough guide, expect **10–40 min per run with
ablations**, so roughly **15–50 GPU-VM hours** for the grid when 3 workers share one VM.

Rough prices at the time of writing (check the [pricing calculator](https://cloud.google.com/products/calculator)):
`g2-standard-16` + 1× L4 costs about US$1–1.5 per hour on demand, and Spot is roughly a third of that.
The MIMIC-III BigQuery query scans a few tens of GB, which is free within the 1 TB monthly free
tier. A 200 GB balanced disk costs about US$20 per month while the VM exists.

**Spot VMs** are fine here: every run writes its own `metrics.json`, so after a preemption you
restart the VM, rerun `gcp/run_grid.sh`, and finished runs are skipped.

---

## 3. One-time Google Cloud setup (laptop)

```bash
# 3.1 Install the gcloud CLI: https://cloud.google.com/sdk/docs/install , then:
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable compute.googleapis.com bigquery.googleapis.com storage.googleapis.com

# 3.2 GPU quota. Console → IAM & Admin → Quotas, and request:
#     "GPUs (all regions)" ≥ 1   and   "NVIDIA L4 GPUs" ≥ 1 in your region (e.g. us-central1)
#     New projects often have 0 GPU quota; approval can take from minutes to a day.

# 3.3 Find a zone with L4 GPUs
gcloud compute accelerator-types list --filter="name=nvidia-l4" --format="value(zone)" | sort | head
```

If L4 is unavailable, use `MACHINE=n1-standard-16` plus a T4. In `gcp/create_vm.sh`, add
`--accelerator=type=nvidia-tesla-t4,count=1` to the create command. The models are small, so a
T4 is enough.

---

## 4. Put the code somewhere the VM can reach

Recommended: a **private** GitHub repository (the code contains no data).

```bash
cd dp-ctgan
git add -A
git commit -m "Experiment pipeline for TIST revision"
git remote set-url origin git@github.com:YOUR_USER/dp-ctgan.git   # or create a new private repo
git push
```

Alternative without GitHub: `gcloud compute scp --recurse ./dp-ctgan dpctgan-gpu:~ --zone ZONE`
(run after step 5).

---

## 5. Create the VM (laptop)

```bash
cd dp-ctgan
PROJECT=YOUR_PROJECT_ID ZONE=us-central1-a SPOT=true bash gcp/create_vm.sh
gcloud compute scp gcp/setup_vm.sh dpctgan-gpu:~ --zone us-central1-a
gcloud compute ssh dpctgan-gpu --zone us-central1-a
```

---

## 6. Set up the VM (on the VM)

```bash
REPO_URL=https://github.com/YOUR_USER/dp-ctgan.git bash setup_vm.sh   # installs NVIDIA driver, may reboot
# after the reboot, ssh in again and run it a second time:
REPO_URL=https://github.com/YOUR_USER/dp-ctgan.git bash setup_vm.sh
cd ~/dp-ctgan && source .venv/bin/activate
nvidia-smi        # should list the L4
```

For a private GitHub repo, either add an SSH deploy key on the VM (`ssh-keygen`, then add the
public key under the repo's *Settings → Deploy keys*) or use an HTTPS personal access token.

> Every later step assumes you are in `~/dp-ctgan` with `.venv` activated.

---

## 7. Prepare the datasets (on the VM)

### 7.1 Public datasets

```bash
# Optional but preferred for Credit Card Fraud: Kaggle API token
#   kaggle.com → Settings → API → "Create New Token" → upload kaggle.json to the VM:
#   (laptop) gcloud compute scp ~/Downloads/kaggle.json dpctgan-gpu:~ --zone ZONE
mkdir -p ~/.kaggle && mv ~/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

python data/prepare_public_datasets.py
# -> data/adult.csv (48,842 rows; 45,222 after removing missing values)
# -> data/UCI_Credit_Card.csv (30,000 rows, 22.12% default)
# -> data/creditcard.csv (284,807 rows, 492 frauds; falls back to OpenML if Kaggle is not set up)
```

### 7.2 MIMIC-III from BigQuery

**Access.** Your PhysioNet account must have MIMIC-III credentialed access, and the Google account
you linked at <https://physionet.org/settings/cloud/> must be able to query `physionet-data`.
Check this in the BigQuery console first: the `physionet-data` project should list
`mimiciii_clinical`.

**Credentials on the VM.** The VM's service account has no PhysioNet access, so log in as
**yourself**:

```bash
gcloud auth application-default login --no-launch-browser
# open the printed URL on your laptop, sign in with the PhysioNet-linked Google account, paste the code back
```

**Run the extraction.**

```bash
python data/prepare_mimic3.py --project YOUR_PROJECT_ID --dry-run   # prints GB scanned and the cost estimate
python data/prepare_mimic3.py --project YOUR_PROJECT_ID
# -> data/mimic3.csv : about 50,000 adult admissions with an ICU stay, about 11–12% in-hospital mortality
```

**About the cohort (`data/mimic3_cohort.sql`).** There is one row per adult admission with at
least one ICU stay. Features:

- age (capped at 90), gender, admission type, insurance, marital status, first care unit;
- hospital and ICU length of stay, number of ICU stays, diagnoses and procedures;
- heart rate, SpO₂ and temperature min/mean/max over the **first 24 h of the first ICU stay**.

The target is `hospital_expire_flag`. Copy this description into the manuscript, which resolves
the MIMIC cohort TODO. If you still have the original `final_dataset.csv` from the thesis, you may
use it instead (`cp final_dataset.csv data/mimic3.csv`), but **all** MIMIC numbers must then come
from one cohort.

> Length of stay and diagnosis counts are only known at discharge. That is acceptable for
> evaluating synthetic-data utility, but say so in the paper. To make the task strictly
> "admission-time", drop those columns via `id_cols` in the `mimic3` preset.

**Data governance (PhysioNet DUA):**
- keep `data/mimic3.csv` and `results/mimic3/**/synthetic_*.csv.gz` only on the VM, your laptop,
  or a **private** bucket;
- never commit them (`.gitignore` already excludes them);
- do not share or upload synthetic MIMIC data.

---

## 8. Smoke test, then a timing run (on the VM)

```bash
# 8.1 Smoke test: tiny settings, ~2–5 minutes, checks that everything runs end to end
python experiments/run_experiments.py --datasets adult --eps 3 --seeds 0 --quick --ablations --out results_smoke
cat results_smoke/adult/eps3/seed0/metrics.json | head -60
rm -rf results_smoke

# 8.2 Timing run: one real run per dataset at ε = 3 (seed 0)
python experiments/run_experiments.py --eps 3 --seeds 0 --ablations
grep -h '"total"' results/*/eps3/seed0/metrics.json
```

Use the measured `total` seconds to estimate the grid: `sum over datasets × 4 budgets × 5 seeds ÷
number of workers`. **Keep these four results**: the full grid skips them.

If a dataset is too slow:
- lower `rare_max_iters` or `rare_pool_size` in `HP_PRESETS`;
- drop `--ablations` for ε = 1 and ε = 10 (the paper reports ablations at one budget, ε = 3 by default);
- use 3 seeds instead of 5 (state this in the paper).

---

## 9. Hyperparameter tuning (optional but recommended; on the VM)

```bash
tmux new -s tune
python experiments/tune.py --trials 20            # all datasets × {No-DP, 1, 3, 10}; writes results/hparams/*.json
# Ctrl-b d to detach; tmux attach -t tune to come back
```

- Tuning uses only the seed-0 training and validation splits. The test set and the final MIA
  holdout are never used.
- `run_experiments.py` loads `results/hparams/<dataset>_eps<e>.json` automatically.
- **Tune before the full grid.** If you tuned after the timing run, delete those four timing
  results so they are recomputed with the tuned settings.
- The paper must say that tuning on the private data is **not** included in ε (already worded in
  the revised manuscript). Update Appendix C: this code tunes one stage (epochs, learning rates,
  batch size, and clipping norm or R1 weight), not the three stages described in the thesis.

---

## 10. The full grid (on the VM)

```bash
bash gcp/run_grid.sh 3                # 3 parallel workers, all datasets × {No-DP,1,3,10} × seeds 0–4, with ablations
tmux attach -t grid                   # watch; Ctrl-b n / p switches worker windows; Ctrl-b d detaches

# progress from another terminal
find results -name metrics.json | wc -l          # completed runs (target: 80)
find results -name error.txt                     # failed runs (read them, fix, delete error.txt, rerun)
nvidia-smi; htop
```

Variants:

```bash
EXTRA="--ablations --eps 3" bash gcp/run_grid.sh 3              # ablations only at ε = 3 first
EXTRA="--eps none 1 10" bash gcp/run_grid.sh 3                  # remaining budgets without ablations
EXTRA="--ablations --seeds 0 1 2" bash gcp/run_grid.sh 2        # 3 seeds, 2 workers
```

**After a Spot preemption:** start the VM (`gcloud compute instances start dpctgan-gpu --zone
ZONE`), SSH in, `cd ~/dp-ctgan && source .venv/bin/activate && bash gcp/run_grid.sh 3`. Finished
runs are skipped. A run that was interrupted mid-way restarts from the beginning.

**Back up regularly** to a private bucket:

```bash
gcloud storage buckets create gs://YOUR_BUCKET --location=us-central1 --uniform-bucket-level-access   # once
BUCKET=gs://YOUR_BUCKET bash gcp/sync_results.sh
```

---

## 11. Baselines (on the VM, separate environments)

Baseline libraries pin different versions of torch and opacus, so give each one its own
environment. They share the same splits and evaluation (`experiments/common.py`).

```bash
# 11.1 synthcity: PATE-GAN, DP-GAN, AIM, PrivBayes (DP) + CTGAN, TVAE (non-private references)
uv venv --python 3.11 .venv-synthcity && source .venv-synthcity/bin/activate
uv pip install synthcity pandas scikit-learn scipy
python experiments/run_baselines.py --methods pategan dpgan aim --eps 1 3 10
python experiments/run_baselines.py --methods ctgan tvae --eps none
deactivate

# 11.2 SMOTE on real data (non-private utility reference; no privacy metrics)
source .venv/bin/activate          # imbalanced-learn is already installed by setup_vm.sh
python experiments/run_baselines.py --methods smote --eps none

# 11.3 optional: SmartNoise DP-CTGAN / PATE-CTGAN / MST
uv venv --python 3.11 .venv-sn && source .venv-sn/bin/activate
uv pip install smartnoise-synth scikit-learn pandas scipy
python experiments/run_baselines.py --methods sn_patectgan sn_mst --eps 1 3 10
deactivate
```

Notes:
- Run the baselines in tmux too; synthcity's PATE-GAN and AIM can take a long time on 100k-row
  datasets. Use `--datasets` and `--seeds 0 1 2` to stage the work.
- If a plugin fails, read `results/baselines/<method>/.../error.txt`. Library APIs change between
  versions; the generator calls live in `gen_synthcity()` and `gen_smartnoise()`.
- DP-CGAN (`dp-cgans`) has no accountant, so it is not run here. The revised manuscript keeps it
  only in the appendix. Drop it unless a reviewer asks.

---

## 12. Aggregate, make figures, bring results home

```bash
source .venv/bin/activate
python experiments/aggregate.py --results results --ablation-eps 3
python experiments/make_figures.py --results results --fig-eps 3
ls results/summary results/figures
BUCKET=gs://YOUR_BUCKET bash gcp/sync_results.sh
```

On your laptop:

```bash
gcloud storage rsync --recursive gs://YOUR_BUCKET/results/summary ./results/summary
gcloud storage rsync --recursive gs://YOUR_BUCKET/results/figures ./results/figures
```

**Then delete the VM** (the disk is billed while it exists):

```bash
gcloud compute instances delete dpctgan-gpu --zone us-central1-a
```

---

## 13. Visualisations: what to put in the paper

The thesis had no result figures. TIST reviewers expect a few figures that carry the argument,
not decoration. `make_figures.py` produces all of the following. They use one colour-blind-safe
palette (blue = combined, orange = common only, grey = real data), and every series also has its
own marker or line style, so they survive greyscale printing.

**Main paper (5 figures + the existing 3 method diagrams):**

| Figure | File | Story it tells | Where |
|---|---|---|---|
| Privacy–utility curves | `fig_privacy_utility.pdf` | RF AUPRC vs ε for common vs combined, against the real-data line; answers RQ2 | Section 6.2 (replaces the "privacy budget" paragraph TODO) |
| Effect sizes | `fig_gains.pdf` | combined − common with 95% CIs per dataset and ε; answers RQ1 with uncertainty | Section 6.1, next to Table *utility* |
| Ablation | `fig_ablation.pdf` | which stage matters, and whether gains exceed plain oversampling | Section 6.4 (with or instead of Table *ablation*) |
| MIA ROC (log FPR) | `fig_mia_roc_domias.pdf` | attack success at low FPR, common vs combined; answers RQ3 | Section 6.3 |
| Sampler diagnostics | `fig_sampler_scores.pdf` | flow-score distribution: pool → refined → selected; shows Stages 2, 3 and 5 do what they claim | Section 4.4 or 6.4 |

**Supplementary / appendix:**

| Figure | File | Use |
|---|---|---|
| PR curves | `fig_pr_curves.pdf` | complements AUPRC numbers |
| Minority marginals | `fig_marginals_<dataset>.pdf` | real vs common vs rare-stream distributions of the 4 most class-separating features |
| t-SNE | `fig_embedding_<dataset>.pdf` | coverage of the real minority region by each stream (qualitative only) |
| Correlation-difference heatmaps | `fig_corrdiff_<dataset>.pdf` | pairwise structure preserved by common vs combined |

LaTeX snippet (figures go in `paper/figures/`):

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig_privacy_utility.pdf}
  \caption{Random-forest AUPRC on the sealed test set as the DP-SGD budget varies (mean $\pm$ std over
  five seeds). Dotted grey: classifier trained on real data.}
  \label{fig:privacy-utility}
\end{figure}
```

Notes:
- Choose the ε for seed-level figures with `--fig-eps` (3 matches the ablation).
- The MIMIC-III figures show synthetic distributions only, no real records. Marginal histograms of
  *real* MIMIC data are aggregate statistics; PhysioNet allows publishing aggregate results, but
  avoid bins with very few patients.
- Do not report t-SNE distances as evidence; use it only as an illustration.

---

## 14. Updating the manuscript from the outputs

1. Replace Table *utility* and Table *fidelity* with `tab_utility.tex` and `tab_fidelity_privacy.tex`.
   Remove "Single run" and the F1 threshold TODO; the captions now say mean ± std and validation thresholds.
2. Fill Table *ablation* from `tab_ablation.tex`. Interpret it honestly: if "plain oversampling" is
   close to "full sampler", the gain comes from rebalancing. Say so, and keep the sampler's
   contribution claims to what the table shows.
3. Replace the PATE-GAN table and the DP-CGAN appendix with `tab_baselines.tex`.
4. Fill the δ/σ/q/T/achieved-ε row from `tab_dp.tex`, and add `tab_gate.tex` to Section 4.5.
5. Replace Appendix D with `tab_confusion.tex` (validation thresholds), and drop the "test-optimal" caveat.
6. Rewrite every number in the abstract, Section 6 and the conclusion from the new tables. Cite
   `stats.txt` for "combined beats common" claims (e.g. "Wilcoxon p < 0.001 over 80 paired runs").
7. Update Section 5 to match this protocol:
   - splits 64/8/8/20;
   - k = 5;
   - δ = min(10⁻⁵, 1/(10 n));
   - DOMIAS-style and DCR metrics added;
   - single-stage tuning;
   - the MIMIC cohort definition;
   - Credit Card Fraud subsampled to 100,000 rows.
8. Report hardware (L4 GPU, vCPUs), software versions (`pip freeze > results/versions.txt`), and
   runtimes (`timings_sec` in each `metrics.json`; `total_min` in `runs.csv`).
9. In the data-availability statement, give the repository URL with a release tag and a Zenodo DOI.
   Archive the code, not the data.

---

## 15. Troubleshooting

| Symptom | Fix |
|---|---|
| `Quota 'GPUS_ALL_REGIONS' exceeded` | Request the quota in section 3.2, or try another region |
| `ZONE_RESOURCE_POOL_EXHAUSTED` | Try another zone that offers the GPU, or Spot later |
| `torch.cuda.is_available()` is False | `nvidia-smi` must work; rerun `setup_vm.sh`; reboot |
| BigQuery `Access Denied: physionet-data` | Wrong Google account in ADC; redo `gcloud auth application-default login` with the PhysioNet-linked account; check PhysioNet cloud settings |
| BigQuery "Unrecognized name: error" | Remove the `AND (ce.error IS NULL OR ce.error = 0)` line in `mimic3_cohort.sql` |
| `Dataset not found at 'data/...'` | Run from the repo root, or pass `--data-dir` |
| Opacus error about the target epsilon being too small | ε = 1 with few epochs may need a larger batch or more data; lower `gan_epochs` or skip ε = 1 for that dataset and say so |
| Out of RAM with 3 workers | Use 2 workers or a larger machine (`g2-standard-24`) |
| A run is stuck in rare sampling | Lower `rare_max_iters` / `rare_pool_size` in `HP_PRESETS` for that dataset |
| `error.txt` in a run folder | Read it, fix, delete the folder, rerun the same `run_grid.sh` command |
