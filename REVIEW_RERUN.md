# Review re-run: what changed in the code and how to run it (VM)

This round answers the internal review (see `paper/RareSel-DP_Internal_Review_Assessment.docx`). It adds
three sets of runs to the round-2 outputs; nothing else in `results_dp/` or `results/` is re-run.

| Step | Review point | Runs | Writes to | Time (4 × T4) |
|---|---|---|---|---|
| A (required) | R10: the rare stream could be shorter than n_r | Credit Card Fraud, ε ∈ {1, 3}, seeds 0–4, with ablations (10 runs) | `results_dp/credit_fraud/eps1, eps3` | ~30 min |
| B (recommended) | R21: baseline parity | SmartNoise DP-CTGAN, PATE-CTGAN, MST, matched δ and public bounds, seeds 0–4 (180 runs) | `results/baselines/sn_*_matched/` | ~3–4 h |
| C (recommended) | R3: tuning leakage and tuning asymmetry | RareSel-DP with the dataset presets (no tuned file), no ablations (80 runs) | `results_dp_untuned/` | ~2 h |

B and C run at the same time, after A. Total wall-clock about 4–5 hours.

---

## 1. What changed in the code

**`dp_ctgan_pipeline.py` — `sample_rare` (R10).** Before: when fewer than n_r candidates passed the judge
within `rare_max_iters` rounds, the sampler returned only those survivors (Credit Card Fraud ε = 3 seed 0:
1 record instead of 474); with zero survivors it returned n_r plain samples without diagnostics. Now it
always returns n_r records: the judge-selected ones, completed with plain G(z, c_min) samples (still
post-processing, no privacy cost). It records `n_survivors`, `n_selected` and `n_padded`.

Why all 10 Credit Card Fraud runs at ε ∈ {1, 3}: in every one of them the sampler used all 30 rounds
(~54 s against ~2 s in every other run), so the judge accepted fewer than 2·n_r candidates, and in at least
one (ε = 1 seed 0) it accepted none. The old outputs do not record how many records the judge selected; the
re-run does. All other 70 runs left the loop early, which means the judge accepted at least 2·n_r candidates,
so they are not affected.

**`experiments/run_experiments.py`.** Stores `meta.rare_sampler = {n_survivors, n_selected, n_padded}` and
the same counts in every ablation block; `--hp-dir none` runs the dataset presets; `meta.hp_source` records
which hyperparameters a run used.

**`experiments/run_baselines.py` (R21, X6).** `--matched` gives the SmartNoise baselines the same treatment
as RareSel-DP: δ = min(10⁻⁵, 1/(10 n_train)) passed to the library, and the training-range column bounds
treated as public (as RareSel-DP's encoder and clipping ranges), so `preprocessor_eps = 0` and the full ε
goes to the generator. Results go to `<method>_matched/`, so the round-2 runs (library defaults) stay as
they are. `meta.delta` now records the δ the generator actually used (round 2 stored RareSel-DP's δ,
which was never passed to SmartNoise), and `meta.smartnoise` the preprocessing settings.

**`experiments/aggregate.py`.** Reads both SmartNoise variants (labels "(SmartNoise)" and "(SmartNoise,
matched)"); `runs.csv` gains `delta_used`, `sn_preprocessor_eps`, `hp_source` and `rare_n_selected` /
`rare_n_padded`; for the round-2 SmartNoise runs it reports the library's own δ. `stats.txt` gains the
review's tests: DP runs only, without seed 0 (the tuning seed), dataset–seed units, per-dataset Holm
p-values with the number of seeds with a positive gain, and the list of padded rare streams.

**`experiments/make_figures.py`.** `fig_baselines` draws one SmartNoise variant (`--sn-variant`, default
`matched` when those runs exist). Seed-level figures skip a dataset whose judge selected fewer than half of
the n_r records (the script prints which).

**`experiments/check_outputs.py`.** Checks the matched baselines (5 seeds), the padding fix on Credit Card
Fraud, and the untuned grid (`--hp-dir none`).

**Docs.** The pipeline header and `README.md` now describe RareSel-DP and the private pipeline; the old
header called it "RareFlow-DP" and said the data were "NOT end-to-end DP" (true only for `--mode legacy`).
The standalone CLI (`python dp_ctgan_pipeline.py ...`) now defaults to `--mode dp`; the experiment scripts
already set the mode explicitly, so no reported number depends on this.

---

## 2. Get the code onto the VM

On your laptop (the repository is `bikrant_thesis/dp-ctgan`):

```bash
cd ~/Downloads/bikrant_thesis/dp-ctgan
git add README.md REVIEW_RERUN.md dp_ctgan_pipeline.py experiments/
git commit -m "Review fixes: pad the rare stream to n_r, matched SmartNoise baselines, untuned grid, docs"
git push
```

On the VM:

```bash
cd ~/projects/dp-ctgan
git pull
source .venv/bin/activate                 # main environment (torch 2.2, opacus 1.5.3)
mkdir -p logs
tar czf ~/backup_before_review_$(date +%F).tgz results_dp results/baselines results/hparams   # ~120 MB
```

---

## 3. Smoke tests (~10 minutes)

**Sampler fix** (tiny settings; the short pool makes padding likely, which is what we want to see):

```bash
python3 experiments/run_experiments.py --datasets credit_fraud --eps 3 --seeds 0 --quick --ablations \
    --out results_smoke_review
ls results_smoke_review/credit_fraud/eps3/seed0/error.txt 2>/dev/null      # must print nothing
python3 -c "
import json; R = json.load(open('results_smoke_review/credit_fraud/eps3/seed0/metrics.json'))
print('main     ', R['meta']['n_rare'], R['meta']['rare_sampler'])
for k, v in R['ablations'].items(): print(f'{k:17s}', v['n_rare'], {x: v.get(x) for x in ('n_selected', 'n_padded')})"
rm -rf results_smoke_review
```

Expected: every line has the same `n_rare`, and `n_selected + n_padded = n_rare` (`plain_oversample`
has no counts).

**Matched SmartNoise** (Taiwan is the fastest dataset; these runs are kept, they are part of Step B):

```bash
for m in sn_mst sn_patectgan sn_dpctgan; do
  .venv-sn/bin/python experiments/run_baselines.py --methods $m --matched --datasets taiwan_credit --eps 3 --seeds 0 \
      --out results/baselines 2>&1 | tee -a logs/review_sn_smoke.log
done
cat results/baselines/sn_*_matched/taiwan_credit/eps3/seed0/error.txt 2>/dev/null     # must print nothing
grep -c "on preprocessor" logs/review_sn_smoke.log                                    # must print 0
.venv-sn/bin/python -c "
import json, glob
for f in sorted(glob.glob('results/baselines/sn_*_matched/taiwan_credit/eps3/seed0/metrics.json')):
    m = json.load(open(f))['meta']; print(m['method'], m['delta'], m['smartnoise'], round(m['fit_seconds'] / 60, 1), 'min')"
```

Expected: δ = 5.21e-06 for all three, `preprocessor_eps: 0.0`, no "Spent … epsilon on preprocessor" line.
If a method fails, send me its `error.txt`; delete its `sn_*_matched/` folder before re-running.

---

## 4. Step A — Credit Card Fraud ε ∈ {1, 3} with the fixed sampler (required)

```bash
mkdir -p results_dp_superseded/credit_fraud
mv results_dp/credit_fraud/eps1 results_dp/credit_fraud/eps3 results_dp_superseded/credit_fraud/
for i in 0 1 2 3; do
  nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=5 LOKY_MAX_CPU_COUNT=5 \
    python3 experiments/run_experiments.py --mode dp --datasets credit_fraud --eps 1 3 --seeds 0 1 2 3 4 \
      --ablations --hp-dir results/hparams --out results_dp --shard $i/4 > logs/review_ccf_$i.log 2>&1 &
done
```

Check (target 10, no errors):

```bash
ls results_dp/credit_fraud/eps{1,3}/seed*/metrics.json 2>/dev/null | wc -l
ls results_dp/credit_fraud/eps{1,3}/seed*/error.txt 2>/dev/null
grep -h "rare stream" logs/review_ccf_*.log          # n_selected / n_padded of each run
```

The old runs stay in `results_dp_superseded/` (outside the aggregation path) for comparison.

---

## 5. Steps B and C — start together once Step A has finished

**B. Matched SmartNoise baselines**, 11 processes (DP-CTGAN on Credit Card Fraud is the slowest, ~13 min
per run, so it is split by budget). The Taiwan ε = 3 seed-0 runs from the smoke test are skipped.

```bash
SN="nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2 .venv-sn/bin/python experiments/run_baselines.py --matched --seeds 0 1 2 3 4 --out results/baselines"
for ds in adult taiwan_credit mimic3; do
  $SN --methods sn_dpctgan --datasets $ds --eps 1 3 10 > logs/review_sn_dpctgan_$ds.log 2>&1 &
done
for e in 1 3 10; do
  $SN --methods sn_dpctgan --datasets credit_fraud --eps $e > logs/review_sn_dpctgan_credit_fraud_eps$e.log 2>&1 &
done
for ds in adult taiwan_credit credit_fraud mimic3; do
  $SN --methods sn_patectgan --datasets $ds --eps 1 3 10 > logs/review_sn_patectgan_$ds.log 2>&1 &
done
$SN --methods sn_mst --eps 1 3 10 > logs/review_sn_mst.log 2>&1 &
```

**C. Untuned RareSel-DP grid** (dataset presets, no ablations, 80 runs):

```bash
for i in 0 1 2 3; do
  nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=4 \
    python3 experiments/run_experiments.py --mode dp --hp-dir none --out results_dp_untuned --shard $i/4 \
      > logs/review_untuned_$i.log 2>&1 &
done
```

**Monitor** (re-run any time):

```bash
for m in sn_dpctgan sn_patectgan sn_mst; do
  echo "$m matched: $(ls results/baselines/${m}_matched/*/eps*/seed*/metrics.json 2>/dev/null | wc -l)/60," \
       "$(ls results/baselines/${m}_matched/*/eps*/seed*/error.txt 2>/dev/null | wc -l) errors"
done
echo "untuned: $(ls results_dp_untuned/*/eps*/seed*/metrics.json 2>/dev/null | wc -l)/80," \
     "$(ls results_dp_untuned/*/eps*/seed*/error.txt 2>/dev/null | wc -l) errors"
grep -l "on preprocessor" logs/review_sn_*.log        # must print nothing
pgrep -fa "run_baselines.py|run_experiments.py" | wc -l   # 0 = all finished
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

Everything is resumable: after a preemption or a crash, delete the `error.txt` files of the failed runs
and start the same loops again; finished runs are skipped.

---

## 6. Tables, figures and checks (after B and C)

```bash
source .venv/bin/activate
python3 experiments/aggregate.py --results results_dp --baselines-dir results/baselines \
    --legacy-results results --ablation-eps 3 > logs/review_aggregate.log 2>&1
rm -rf results_dp/figures
python3 experiments/make_figures.py --results results_dp --fig-eps 3 | tee logs/review_figures.log
python3 experiments/aggregate.py --results results_dp_untuned --ablation-eps 3 > logs/review_aggregate_untuned.log 2>&1
python3 experiments/check_outputs.py --results results_dp --baselines-dir results/baselines | tee logs/review_check.log
python3 experiments/check_outputs.py --results results_dp_untuned --hp-dir none \
    --baselines-dir results/baselines | tee logs/review_check_untuned.log
sed -n '/DP runs only/,$p' results_dp/summary/stats.txt
```

What to look for:

- `check_outputs`: 80/80 runs, the Credit Card Fraud padding check `[ok]`, `sn_*_matched 60/60` each;
  for the untuned grid 80/80 and "every run used its preset hyperparameters".
- `stats.txt`, section "Rare streams completed with plain samples": the runs where the private judge
  accepted fewer than n_r candidates, with the counts (the paper will report these).
- `review_figures.log`: which dataset, if any, is skipped in the seed-level figures.

---

## 7. Bring the outputs home

```bash
cd ~/projects/dp-ctgan
tar czf ~/review_outputs_$(date +%F).tgz results_dp results_dp_untuned results_dp_superseded \
    results/baselines results/hparams logs/review_*
ls -lh ~/review_outputs_*.tgz
```

Download the archive the same way as the v1.0 outputs (for example
`gcloud compute scp VM_NAME:~/review_outputs_*.tgz . --zone ZONE`) and unpack it into a new folder next to
`v1.0_figures_Code_files`, e.g. `bikrant_thesis/v1.1_review_outputs/`. The archive contains synthetic
MIMIC-III records (`results_dp*/mimic3/eps*/seed0/synthetic_*.csv.gz`): keep it private (PhysioNet DUA).

Then the manuscript is updated in one pass from these outputs (tables via `paper/make_paper_tables.py`,
which will also be added to `experiments/` so the repository regenerates every table).
