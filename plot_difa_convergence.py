"""Figure 4: DIFA-2.2 convergence dynamics.

Renders the Label-Predictor / Domain-Discriminator loss curves over the
50-epoch DIFA-2.2 training run, with the ln(2) ≈ 0.693 maximum-confusion
reference line that the manuscript Section 4.4 narrative anchors on.

Reads (in priority order):
    1. models/sedika_difa_v2_history.csv  -- new structured history written
       by the patched sedika_difa_v2.train_difa_v2()
    2. Most recent results/_difa_rerun*.log -- parses the epoch print lines
       as a fallback when the history CSV is absent.

Run:
    python plot_difa_convergence.py
"""
from __future__ import annotations

import glob
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paths import MODEL_DIR, OUTPUT_DIR, PLOT_DIR, ensure_dirs


HISTORY_CSV = os.path.join(MODEL_DIR, "sedika_difa_v2_history.csv")


def _from_csv() -> pd.DataFrame | None:
    if not os.path.exists(HISTORY_CSV):
        return None
    df = pd.read_csv(HISTORY_CSV)
    return df


def _from_log() -> pd.DataFrame | None:
    """Parse the most recent DIFA log file's epoch print lines."""
    logs = sorted(glob.glob(os.path.join(OUTPUT_DIR, "_difa_rerun*.log")),
                  key=os.path.getmtime, reverse=True)
    if not logs:
        return None

    # Match the verbose print line:
    #   Epoch  10 | a_cls=1.00 lam_dann=0.00 g_ent=0.50 | task=0.1234 domain=0.5678 entropy=2.3456
    pat = re.compile(
        r"Epoch\s+(\d+)\s*\|\s*"
        r"a_cls=([\d.]+)\s+lam_dann=([\d.]+)\s+g_ent=([\d.]+)\s*\|\s*"
        r"task=([\d.]+)\s+domain=([\d.]+)\s+entropy=([-\d.]+)"
    )
    rows = []
    for log_path in logs:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    rows.append({
                        "epoch":         int(m.group(1)),
                        "alpha_cls":     float(m.group(2)),
                        "lambda_dann":   float(m.group(3)),
                        "gamma_entropy": float(m.group(4)),
                        "task_loss":     float(m.group(5)),
                        "domain_loss":   float(m.group(6)),
                        "entropy_loss":  float(m.group(7)),
                    })
        if rows:
            return pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    return None


def main():
    ensure_dirs(PLOT_DIR)

    df = _from_csv() or _from_log()
    if df is None or df.empty:
        sys.exit(
            "No DIFA convergence data found.\n"
            f"Expected one of:\n"
            f"  - {HISTORY_CSV}  (written by patched train_difa_v2)\n"
            f"  - results/_difa_rerun*.log  (parsed verbose prints)\n"
            "Run sedika_difa_v2.py to produce one of these."
        )
    print(f"Loaded {len(df)} epochs of DIFA history.")

    fig, ax = plt.subplots(figsize=(11, 6))

    # Task and domain losses
    ax.plot(df["epoch"], df["task_loss"], color="#2ca02c",
            linewidth=2, marker="o", markersize=4,
            label="Label-Predictor loss  (task: target-domain CE)")
    ax.plot(df["epoch"], df["domain_loss"], color="#d62728",
            linewidth=2, marker="s", markersize=4,
            label="Domain-Discriminator loss  (BCE on source/target)")

    # Entropy on a secondary axis if it has signal
    if "entropy_loss" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df["epoch"], df["entropy_loss"], color="#1f77b4",
                 linewidth=1.4, linestyle="--", alpha=0.7,
                 label="Target-entropy minimisation term")
        ax2.set_ylabel("Entropy loss", color="#1f77b4")

    # ln(2) reference line — the maximum-confusion equilibrium claim
    ax.axhline(math.log(2), linestyle=":", linewidth=1.4, color="#444",
               label=r"ln(2) ≈ 0.693 (max-confusion equilibrium for binary discriminator)")

    # Phase annotations
    warmup_end = int(df.loc[df["lambda_dann"] > 0, "epoch"].min()) if (df["lambda_dann"] > 0).any() else None
    if warmup_end is not None:
        ax.axvspan(0, warmup_end - 0.5, color="#f0e68c", alpha=0.30, zorder=0)
        ax.text(warmup_end / 2, 0.05, f"λ = 0 warm-up\n(epoch 1–{warmup_end - 1})",
                fontsize=9, ha="center", color="#665520")

    ax.set_title("DIFA-2.2 adversarial convergence: Label vs Domain vs Entropy loss "
                 "across 50 epochs", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xlim(0.5, df["epoch"].max() + 0.5)
    ax.grid(True, linestyle="--", alpha=0.45)

    # Combine legends from both axes
    lines, labels = ax.get_legend_handles_labels()
    if "entropy_loss" in df.columns:
        l2, lb2 = ax2.get_legend_handles_labels()
        lines += l2; labels += lb2
    ax.legend(lines, labels, loc="upper right", fontsize=9, frameon=True)

    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, "figure4_difa_convergence.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
