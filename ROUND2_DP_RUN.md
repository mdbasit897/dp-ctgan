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
| PATE-GAN | **Yes, after a probe** | ε = 1 and ε = 3 gave bit-identical outputs (training stopped after one iteration) |
| DP-GAN (synthcity) | **Replace** | it collapsed to one label (95–100% minority). Use SmartNoise DP-CTGAN instead |
| AIM | **No** | dropped (move it out of `results/baselines`, section 5) |
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

### 5.1 Move AIM and the broken DP-GAN out of the baseline folder (kept on disk)

```bash
mkdir -p results/_excluded
mv results/baselines/aim   results/_excluded/aim
mv results/baselines/dpgan results/_excluded/dpgan
```

### 5.2 PATE-GAN: find settings that actually train, then re-run

```bash
mv results/baselines/pategan results/_excluded/pategan_round1
python3 experiments/probe_pategan.py --dataset adult
```

The probe prints synthcity's PATE-GAN parameters, then fits it at ε = 1 and ε = 3 for each candidate
setting and says whether the outputs differ. If none of the default candidates says
`outputs differ (usable)`, pass your own grid using the parameter names it printed, for example:

```bash
python3 experiments/probe_pategan.py --grid '{"lamda": [0.0001, 0.001, 0.01], "n_teachers": [10, 50, 100]}'
```

Pick the setting with different outputs and the best validation AUPRC, then run all datasets with it:

```bash
nohup python3 experiments/run_baselines.py --methods pategan --eps 1 3 10 --seeds 0 1 2 \
    --plugin-kwargs '{"lamda": 0.01}' --out results/baselines > logs/bl_pategan_r2.log 2>&1 &
```

The settings are stored in each `metrics.json` (`meta.plugin_kwargs`); report them in the paper. If no setting
makes ε = 1 and ε = 3 differ, keep the default runs but say in the paper that synthcity's PATE-GAN stops after
its first iteration at ε ≤ 3.

### 5.3 SmartNoise DP-CTGAN, PATE-CTGAN, MST (replaces DP-GAN)

SmartNoise pins its own torch/opacus versions, so give it its own environment:

```bash
uv venv --python 3.11 .venv-sn
uv pip install --python .venv-sn/bin/python smartnoise-synth scikit-learn pandas scipy
.venv-sn/bin/python -c "from snsynth import Synthesizer; print('snsynth ok')"

# test one run first
.venv-sn/bin/python experiments/run_baselines.py --methods sn_dpctgan --datasets adult --eps 3 --seeds 0 \
    --out results/baselines
cat results/baselines/sn_dpctgan/adult/eps3/seed0/error.txt 2>/dev/null   # empty = OK

nohup .venv-sn/bin/python experiments/run_baselines.py --methods sn_dpctgan sn_patectgan sn_mst \
    --eps 1 3 10 --seeds 0 1 2 --out results/baselines > logs/bl_smartnoise.log 2>&1 &
```

If `smartnoise-synth` will not install on Python 3.11, try `--python 3.10`. If one method fails on one
dataset (MST needs every column discretised; it bins continuous columns with a small share of ε), read
`error.txt`; a method that cannot run can be left out, since tables and figures only include methods with results.
These SmartNoise calls were written against the documented API but could not be tested here; check the first
run before launching all of them.

---

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
python3 experiments/make_figures.py --results results_dp --fig-eps 3
cat results_dp/summary/stats.txt
```

New outputs in `results_dp/summary/`:

| File | Content |
|---|---|
| `tab_budget.tex` | ε per component (discriminator, statistics, judge), total by basic and RDP composition, moment noise / n, MM weight, judge noise multiplier |
| `tab_dp_vs_legacy.tex` | RF AUPRC common / combined / gain: round 1 vs privatised pipeline, per dataset and ε |
| `tab_ablation.tex` | now includes "judge = DP discriminator" and "judge = non-private RF (reference)" |
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
