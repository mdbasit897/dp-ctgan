#!/usr/bin/env python3
"""Audit the privacy accounting of the end-to-end protocol and write the per-run privacy ledger.

For every RareSel-DP run (metrics.json) it checks that the run used the end-to-end protocol (private
encoder, dp mode), that the four components add up to at most the target epsilon (basic composition) at
delta/4 each, that the RDP total is not above it either, and that the hyperparameter file was selected
without the run's own dataset (tune_lodo.py, selected_from). For every end-to-end SmartNoise run it checks
that preprocessing and generator budgets add up to the target. Exits with status 1 if any check fails.

  python experiments/audit_privacy.py --results results_e2e --baselines-dir results_e2e/baselines
  python experiments/audit_privacy.py --results results_e2e_untuned
Writes <results>/summary/privacy_ledger.csv (one row per DP run) and prints a per-setting summary.
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

TOL = 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--baselines-dir", default=None)
    ap.add_argument("--allow-encoder-data", action="store_true",
                    help="for the public-encoder ablation grid: do not fail on encoder = data")
    a = ap.parse_args()
    problems, rows = [], []
    files = sorted(glob.glob(os.path.join(a.results, "*", "eps*", "seed*", "metrics.json")))
    files = [f for f in files if os.sep + "baselines" + os.sep not in f]
    if not files:
        sys.exit(f"no runs under {a.results}")
    for f in files:
        m = json.load(open(f))["meta"]
        tag = f"{m['dataset']}/eps{m['epsilon']}/seed{m['seed']}"
        if m.get("protocol") != "e2e" or m.get("mode") != "dp":
            problems.append(f"{tag}: protocol={m.get('protocol')} mode={m.get('mode')}")
        if m.get("encoder") != "dp" and not a.allow_encoder_data:
            problems.append(f"{tag}: encoder={m.get('encoder')}")
        src = m.get("hp_source", "")
        if src.endswith(".json") and os.path.exists(src):
            sel = json.load(open(src)).get("selected_from")
            if sel is None or m["dataset"] in sel:
                problems.append(f"{tag}: hyperparameters {src} not selected without {m['dataset']} ({sel})")
        elif src.endswith(".json"):
            problems.append(f"{tag}: hyperparameter file {src} not found")
        if m["epsilon"] is None:
            continue
        b, enc = m.get("dp_budget") or {}, m.get("encoder_info") or {}
        eps = float(m["epsilon"])
        total, rdp = b.get("epsilon_total_basic", float("inf")), b.get("epsilon_total_rdp", float("inf"))
        if total > eps * (1 + TOL):
            problems.append(f"{tag}: basic total {total:.6f} > {eps}")
        if rdp > eps * (1 + TOL):
            problems.append(f"{tag}: RDP total {rdp:.6f} > {eps}")
        if m.get("encoder") == "dp":
            if not enc.get("private") or abs(b.get("delta_each", 0) * 4 - b.get("delta", -1)) > 1e-15:
                problems.append(f"{tag}: encoder not private or delta not split in four")
        rs = enc.get("range_source", {})
        rows.append({"dataset": m["dataset"], "epsilon": eps, "seed": m["seed"], "n_train": m["n_train"],
                     "delta": b.get("delta"), "delta_each": b.get("delta_each"),
                     "eps_encoder": b.get("epsilon_encoder_achieved"), "eps_discriminator": b.get("epsilon_gan_achieved"),
                     "eps_stats": b.get("epsilon_stats"), "eps_judge": b.get("epsilon_judge_achieved"),
                     "eps_total_basic": total, "eps_total_rdp": rdp,
                     "sigma_encoder": b.get("encoder_noise_multiplier"),
                     "noise_sd_ranges": enc.get("noise_sd_ranges"), "noise_sd_hist": enc.get("noise_sd_hist"),
                     "n_public_bounds": sum(v == "public" for v in rs.values()),
                     "n_private_ranges": sum(v == "private" for v in rs.values()),
                     "sigma_discriminator": (m.get("dp") or {}).get("noise_multiplier"),
                     "q_discriminator": (m.get("dp") or {}).get("sample_rate"),
                     "steps_discriminator": (m.get("dp") or {}).get("steps_done"),
                     "C_discriminator": (m.get("dp") or {}).get("max_grad_norm"),
                     "sigma_stats": (m.get("minority_stats") or {}).get("noise_multiplier"),
                     "sigma_judge": b.get("judge_noise_multiplier"), "steps_judge": b.get("judge_steps"),
                     "hp_source": src})
    print(f"{len(files)} RareSel-DP runs checked, {len(rows)} with DP")
    if a.baselines_dir:
        n_bl = 0
        for f in sorted(glob.glob(os.path.join(a.baselines_dir, "*_e2e", "*", "eps*", "seed*", "metrics.json"))):
            m = json.load(open(f))["meta"]
            sn = m.get("smartnoise") or {}
            tag = f"{m['method']}/{m['dataset']}/eps{m['epsilon']}/seed{m['seed']}"
            n_bl += 1
            if sn.get("protocol") != "e2e":
                problems.append(f"{tag}: not an end-to-end run")
            elif abs(sn["preprocessor_eps"] + sn["epsilon_generator"] - float(m["epsilon"])) > TOL:
                problems.append(f"{tag}: preprocessing + generator epsilon != target")
            elif sn["delta_preprocessor"] + sn["delta_generator"] > m["delta"] * (1 + TOL):
                problems.append(f"{tag}: delta above target")
        print(f"{n_bl} end-to-end SmartNoise runs checked")
    if rows:
        L = pd.DataFrame(rows).sort_values(["dataset", "epsilon", "seed"])
        os.makedirs(os.path.join(a.results, "summary"), exist_ok=True)
        out = os.path.join(a.results, "summary", "privacy_ledger.csv")
        L.to_csv(out, index=False)
        cols = ["eps_encoder", "eps_discriminator", "eps_stats", "eps_judge", "eps_total_basic", "eps_total_rdp",
                "n_private_ranges"]
        print(L.groupby(["dataset", "epsilon"])[cols].mean().round(3).to_string())
        print(f"wrote {out}")
    if problems:
        print(f"\n{len(problems)} PROBLEMS:")
        for p in problems[:50]:
            print("  " + p)
        sys.exit(1)
    print("\nALL PRIVACY CHECKS PASSED")


if __name__ == "__main__":
    main()
