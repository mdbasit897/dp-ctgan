# Round 2: the privatised pipeline (`--mode dp`)

The first round (`results/`) showed that most of the rare-stream gain comes from the Stage-4 random-forest
judge, which is trained on real data **outside** the privacy budget. Round 2 re-runs the main grid with every
component that reads real data covered by ε, so the paper can claim an (ε, δ) guarantee for the released data
(except the GMM encoder, see below).

Nothing in `results/` is overwritten: round 2 writes to `results_dp/`. Your first-round tables, figures and
baselines stay as they are.

---

## 1. What changed in the code

| Component | Round 1 (`--mode legacy`) | Round 2 (`--mode dp`, default) |
|---|---|---|
| Discriminator | DP-SGD at ε | DP-SGD at 0.8 ε |
| Minority mean/std (moment matching, fine-tuning) | exact, from real data | **Gaussian mechanism** at 0.1 ε; down-weighted automatically when noise/n is large; fine-tuning skipped when unreliable |
| Feature matching (FM, MFM), correlation matching | real batches, outside ε | **off** |
| Stage-4 judge | random forest on real data, **not private** | **DP-SGD MLP** at 0.1 ε |
| Stage-6 quantile matching | real minority quantiles, outside ε | **off** (it added no utility in round 1) |
| PrivacyGate | filters | **audit only**: reports near-copy and outlier rates, does not filter |
| Minority rate and number of rare records | exact counts | from the noisy count (no extra budget) |
| Privacy accounting | discriminator only | every component; total by basic composition (≤ ε) and by RDP composition |
| Still non-private | – | the GMM mode encoder and the column ranges used for clipping. State this in the paper; the other DP-CTGAN implementations treat them the same way |

Budget split, δ split: δ/3 each for discriminator, statistics and judge. Change the shares with
`budget_frac_judge` and `budget_frac_stats` in `Hyperparams`. Small shares are certified with an extended RDP
order list (works on Opacus 1.5.3 and 1.6). The one-shot statistics release uses the exact analytic Gaussian
mechanism (Balle & Wang, ICML 2018).

New ablation variants in every run (same GAN, same number of added minority records):

| Variant | Question it answers |
|---|---|
| `plain_oversample` | Is the gain just from adding minority rows? |
| `no_flow_refine`, `no_refine` | Do the flows (Stages 2–3) help when the judge is private? |
| `no_judge` | How much does the private judge contribute? |
| `no_maxmin` | How much does diversity selection contribute? |
| `judge_disc` | Can the DP discriminator replace the trained judge (no extra budget)? |
| `judge_rf_nonprivate` | Reference: the first-round non-private RF judge on the same GAN. The gap between this and the full sampler is the cost of privatising the judge |

`judge_rf_nonprivate` is reported for comparison only; it is not a private method.

---

## 2. What you need to run (and what you do NOT need to re-run)

| Step | Re-run? | Why |
|---|---|---|
| Datasets, MIMIC-III extraction | **No** | unchanged |
| Tuning (`tune.py`) | **No** (optional) | the round-1 settings in `results/hparams/` are reused. Optional re-tune in dp mode: section 6 |
| Main grid, 80 runs with ablations | **Yes** | the GAN, the judge and the statistics all changed, and round 1 did not save trained models |
| SMOTE, CTGAN, TVAE baselines | **No** | non-private references; unaffected |
| PATE-GAN, DP-GAN (synthcity) | **No, dropped** | PATE-GAN: ε = 1 and ε = 3 gave identical outputs; DP-GAN collapsed to one label. Replaced by SmartNoise |
| AIM | **No** | dropped |
| SmartNoise DP-CTGAN / PATE-CTGAN / MST | **New, recommended** | the direct DP competitors |

Compute: about 25% more than round 1 (judge training plus two more ablation variants), so roughly 11
GPU-hours for the grid; with 3–4 workers in parallel, about 3–4 hours wall-clock.

---

## 3. Pull the code and smoke-test (VM)

```bash
cd ~/projects/dp-ctgan && source .venv/bin/activate      # or your (dp-ctgan) environment
git pull
python3 -c "import opacus; print(opacus.__version__)"    # 1.5.3 or 1.6 both work

# smoke test: tiny settings, ~3 minutes, writes to results_smoke_dp/
python3 experiments/run_experiments.py --datasets mimic3 credit_fraud --eps 1 none --seeds 0 \
    --quick --ablations --out results_smoke_dp
ls results_smoke_dp/*/eps*/seed0/error.txt 2>/dev/null     # must print nothing
python3 -c "
import json; R = json.load(open('results_smoke_dp/mimic3/eps1/seed0/metrics.json'))
print(R['meta']['dp_budget']); print(sorted(R['ablations']))"
rm -rf results_smoke_dp
```

Expected: `epsilon_total_basic` ≤ 1, `epsilon_total_rdp` smaller still, and 7 ablation names, including
`judge_disc` and `judge_rf_nonprivate`. The log prints one `privacy:` line per DP run with the split.

---

## 4. Main grid, round 2 (VM)

```bash
mkdir -p logs
for i in 0 1 2 3; do      # 4 workers; each picks its own GPU (experiments/gpu_select.py)
  nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=5 LOKY_MAX_CPU_COUNT=5 \
    python3 experiments/run_experiments.py --mode dp --shard $i/4 --ablations \
      --hp-dir results/hparams --out results_dp > logs/dp_worker_$i.log 2>&1 &
done
wait
```

Monitor and check:

```bash
ls results_dp/*/eps*/seed*/metrics.json 2>/dev/null | wc -l           # target 80
ls results_dp/*/eps*/seed*/error.txt 2>/dev/null                       # failed runs
for f in logs/dp_worker_*.log; do echo "$f: $(grep '^=== ' $f | tail -1)"; done
python3 experiments/check_outputs.py --results results_dp --hp-dir results/hparams --baselines-dir results/baselines
```

Runs are resumable. After a crash or preemption, run the same loop again.

---

## 5. Baselines

Final baseline set: SMOTE, CTGAN, TVAE (non-private references, already done) and SmartNoise DP-CTGAN,
PATE-CTGAN and MST (the DP competitors). synthcity PATE-GAN (ε = 1 and 3 gave identical output), DP-GAN
(collapsed to one class) and AIM are dropped and moved out of the baseline folder (kept on disk).

### 5.1 Move the dropped baselines out (VM, main environment)

```bash
cd ~/projects/dp-ctgan
mkdir -p results/_excluded
mv results/baselines/pategan results/baselines/dpgan results/_excluded/
mv results/baselines/NOTE_aim_credit_fraud.txt results/_excluded/ 2>/dev/null
ls results/baselines        # ctgan  smote  tvae
```

### 5.2 SmartNoise environment (separate; leaves the main environment untouched)

smartnoise-synth 1.0.8 pins opacus 0.14 and pulls a recent torch and NumPy 2, so it must not go into the main
environment. `--torch-backend auto` picks the torch build that matches the VM's NVIDIA driver (download of a
few GB, about 10 minutes).

```bash
uv venv --python 3.11 .venv-sn
uv pip install --python .venv-sn/bin/python --torch-backend auto smartnoise-synth scikit-learn pandas scipy
.venv-sn/bin/python -c "import snsynth, torch; print('snsynth ok, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

If the last line fails with a CUDA/driver error, reinstall with `--torch-backend cpu` (MST is CPU-only anyway;
DP-CTGAN and PATE-CTGAN then run slower but correctly).

### 5.3 Smoke test: one run per method on Adult

```bash
for m in sn_mst sn_dpctgan sn_patectgan; do
  .venv-sn/bin/python experiments/run_baselines.py --methods $m --datasets adult --eps 3 --seeds 0 --out results/baselines
done
ls results/baselines/sn_*/adult/eps3/seed0/                       # each should contain metrics.json
cat results/baselines/sn_*/adult/eps3/seed0/error.txt 2>/dev/null  # must print nothing
```

### 5.4 Full run: 3 methods × 4 datasets × ε ∈ {1, 3, 10} × 3 seeds = 108 runs

One process per method, in parallel. Finished runs (including the smoke tests) are skipped.

```bash
mkdir -p logs
for m in sn_mst sn_dpctgan sn_patectgan; do
  nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 \
    .venv-sn/bin/python experiments/run_baselines.py --methods $m --eps 1 3 10 --seeds 0 1 2 \
      --out results/baselines > logs/bl_$m.log 2>&1 &
done
```

Monitor:

```bash
for m in sn_mst sn_dpctgan sn_patectgan; do
  echo "$m: $(ls results/baselines/$m/*/eps*/seed*/metrics.json 2>/dev/null | wc -l)/36 done," \
       "$(ls results/baselines/$m/*/eps*/seed*/error.txt 2>/dev/null | wc -l) errors," \
       "now: $(grep '^=== ' logs/bl_$m.log | tail -1)"
done
pgrep -fa run_baselines.py | wc -l      # 0 = all finished
```

A failed run leaves `error.txt` and is retried when the loop is started again. Delete the old error files
first (`rm results/baselines/sn_*/*/eps*/seed*/error.txt`). If a method keeps failing on one dataset, it can be
left out: the tables only include runs that have results.

## 6. Optional: re-tune in dp mode

The round-1 settings were tuned for the round-1 objective (with feature matching). They are a reasonable
starting point, so re-tuning is optional. It takes about 3–6 hours:

```bash
for ds in adult taiwan_credit credit_fraud mimic3; do
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=4 \
    python3 experiments/tune.py --mode dp --datasets $ds --trials 20 > logs/tune_dp_$ds.log 2>&1 &
done; wait
# then use --hp-dir results_dp/hparams in section 4
```

---

## 7. Tables and figures (VM)

```bash
python3 experiments/aggregate.py --results results_dp --baselines-dir results/baselines \
    --legacy-results results --ablation-eps 3
rm -rf results_dp/figures          # removes figures that are no longer drawn (sampler scores, MIMIC t-SNE)
python3 experiments/make_figures.py --results results_dp --fig-eps 3
cat results_dp/summary/stats.txt
```

Seed-level figures (PR curves, attack ROC, marginals, correlations, t-SNE) use the seed-0 artefacts and leave
out a dataset whose seed-0 run has a degenerate rare stream; the script prints which one (at ε = 3:
Credit Card Fraud, where the private judge accepted a single candidate). The t-SNE figure leaves out MIMIC-III
by default because it plots real patient records. `fig_sampler_scores` is only drawn with `--sampler-fig`.

New outputs in `results_dp/summary/`:

| File | Content |
|---|---|
| `tab_budget.tex` | ε per component (discriminator, statistics, judge), total by basic and RDP composition, moment noise / n, MM weight, judge noise multiplier |
| `tab_dp_vs_legacy.tex` | RF AUPRC common / combined / gain: round 1 vs privatised pipeline, per dataset and ε |
| `tab_ablation.tex` | now includes "judge = DP discriminator" and "judge = non-private RF (reference)" |
| `tab_baseline_minority.tex` | synthetic minority rate of every baseline |
| `tab_dp.tex` | DP-SGD accounting of the discriminator only (its δ and ε share); totals are in `tab_budget.tex` |
| `stats.txt` | combined vs common, plus every ablation vs the full sampler (paired Wilcoxon) |

All other tables and figures are as in round 1, now in `results_dp/`.

---

## 8. How to read the round-2 results (decides the paper)

Look at `tab_dp_vs_legacy.tex` and the ablation part of `stats.txt`:

1. **Private gain ≥ about half of the round-1 gain** (e.g. MIMIC-III +0.05 or more where round 1 had +0.10 to
   +0.15): the method works privately. The paper is "DP synthetic data with private, classifier-guided,
   diversity-aware minority sampling". Report the round-1 non-private judge only as a reference.
2. **`judge_disc` close to the full sampler**: the discriminator alone is enough, and the judge's 10% of ε can be
   given back to the GAN (`--judge disc`, `budget_frac_judge` then ignored). Simpler method, one fewer component.
3. **`no_flow_refine` still ≈ full sampler**: remove the flows from the method (and from the title); keep them
   as a negative ablation result.
4. **Private gain ≈ 0**: the round-1 gain came from non-private real-data information. That is a finding in
   itself, but it calls for reframing the paper (see the discussion on venue).

Bring `results_dp/summary/`, `results_dp/figures/`, the updated `results/baselines/` and
`results_dp/hparams/` (if re-tuned) back before deleting the VM, as in round 1.
