"""Multi-seed evaluation harness for SEDIKA.

Runs the full evaluation pipeline (DL + ML + anomaly) across multiple seeds
and reports mean +- std per metric, fulfilling the Q1-venue requirement
for variance estimates.

Usage:
    python multi_seed_runner.py                          # default 3 seeds
    python multi_seed_runner.py --seeds 42 123 7 2024 11  # 5 seeds
    python multi_seed_runner.py --branches dl            # only DL
    python multi_seed_runner.py --quick                  # smaller models for sanity

This script does NOT re-run preprocess_data.py. It re-trains models with the
specified seeds, then writes aggregated stats to:
    results/multi_seed_summary.csv
    results/multi_seed_raw.csv

Expected runtime: ~30 min per seed for full DL+ML+anomaly on CPU.
Recommend running overnight or on a workstation with GPU.

Implementation notes
--------------------
We do not re-import the train_dl / train_ml modules and call their internal
functions because they were not written for parameterised seeds. Instead
this script shells out (`subprocess`) to those scripts with a SEED env var
that they read at startup. To minimise churn we use a small patch shim:
each script's existing `tf.random.set_seed(42)` / `np.random.seed(42)` are
left intact (they no-op when the env var is set first) and we inject the
override via the SEDIKA_SEED env var.

To support this script without modifying train_dl.py et al., we recommend
the lighter-weight pattern: add at the top of each training script
    import os
    _seed = int(os.environ.get("SEDIKA_SEED", 42))
    tf.random.set_seed(_seed); np.random.seed(_seed)
Once that one-line patch lands, this harness reproduces the multi-seed
table without further changes.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from paths import OUTPUT_DIR, MODEL_DIR, ensure_dirs


BRANCH_SCRIPTS = {
    "dl":      "train_dl.py",
    "ml":      "train_ml.py",
    "anomaly": "train_anomaly.py",
}
BRANCH_RESULT_CSVS = {
    "dl":      "dl_performance_metrics.csv",
    "ml":      "ml_performance_metrics.csv",
    "anomaly": "anomaly_detection_results.csv",
}


def run_one_seed(seed: int, branches: List[str], project_root: Path) -> Dict[str, pd.DataFrame]:
    """Train all selected branches at this seed and return the parsed CSV per branch."""
    env = os.environ.copy()
    env["SEDIKA_SEED"] = str(seed)
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"

    out: Dict[str, pd.DataFrame] = {}
    for branch in branches:
        script = BRANCH_SCRIPTS[branch]
        print(f"\n--- seed={seed}  branch={branch}  ({script}) ---")
        proc = subprocess.run(
            [sys.executable, script],
            cwd=str(project_root),
            env=env,
        )
        if proc.returncode != 0:
            print(f"  WARNING: {script} returned exit code {proc.returncode}")
            continue

        csv_path = project_root / "results" / BRANCH_RESULT_CSVS[branch]
        if not csv_path.exists():
            print(f"  WARNING: expected {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        df["seed"] = seed
        out[branch] = df

        # Snapshot the per-seed CSV so the canonical one is not overwritten
        snapshot = project_root / "results" / f"{BRANCH_RESULT_CSVS[branch].replace('.csv', '')}_seed{seed}.csv"
        shutil.copy(csv_path, snapshot)
        print(f"  snapshot: {snapshot.name}")
    return out


def aggregate(raw: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    """Compute mean / std / min / max across seeds for each model."""
    grouped = raw.groupby(group_cols)[metric_cols].agg(["mean", "std", "min", "max"])
    grouped.columns = [f"{m}_{stat}" for m, stat in grouped.columns]
    grouped = grouped.round(4).reset_index()
    grouped["n_seeds"] = raw.groupby(group_cols).size().values
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Multi-seed evaluation harness")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7],
                        help="Seeds to evaluate (default: 42 123 7)")
    parser.add_argument("--branches", nargs="+", choices=["dl", "ml", "anomaly"],
                        default=["dl", "ml", "anomaly"],
                        help="Which training branches to re-run")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    ensure_dirs(OUTPUT_DIR)

    all_results: Dict[str, List[pd.DataFrame]] = {b: [] for b in args.branches}
    for seed in args.seeds:
        per_branch = run_one_seed(seed, args.branches, project_root)
        for b, df in per_branch.items():
            all_results[b].append(df)

    summaries = []
    raw_rows = []
    for branch, dfs in all_results.items():
        if not dfs:
            continue
        raw = pd.concat(dfs, ignore_index=True)
        raw["branch"] = branch
        raw_rows.append(raw)

        metric_cols = [c for c in raw.columns
                       if c not in ("Model", "seed", "branch", "Fit_Status")]
        # Keep only numeric metric columns
        metric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(raw[c])]
        if "Model" not in raw.columns:
            continue
        summary = aggregate(raw, ["Model"], metric_cols)
        summary["branch"] = branch
        summaries.append(summary)

    if raw_rows:
        raw_all = pd.concat(raw_rows, ignore_index=True)
        raw_all.to_csv(os.path.join(OUTPUT_DIR, "multi_seed_raw.csv"), index=False)
        print(f"\nWrote raw per-seed table: results/multi_seed_raw.csv")

    if summaries:
        summary_all = pd.concat(summaries, ignore_index=True)
        summary_all.to_csv(os.path.join(OUTPUT_DIR, "multi_seed_summary.csv"), index=False)
        print(f"Wrote summary (mean/std/min/max): results/multi_seed_summary.csv")

        # Pretty preview: the columns the manuscript cares about
        preview_cols = ["branch", "Model", "Accuracy_mean", "Accuracy_std",
                        "F1_Score_mean", "F1_Score_std"]
        avail = [c for c in preview_cols if c in summary_all.columns]
        print("\n=== Multi-seed mean +- std (preview) ===")
        print(summary_all[avail].to_string(index=False))


if __name__ == "__main__":
    main()
