"""Render adversarial_eval.csv as a 2-panel figure: FGSM and PGD degradation
curves, one line per model. Designed as the headline robustness figure for
the thesis.

Usage:
    python plot_adversarial.py
    python plot_adversarial.py --csv results/adversarial_eval.csv \
        --out plots/adversarial_degradation.png
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt

from paths import OUTPUT_DIR, PLOT_DIR, ensure_dirs


# Consistent visual identity per model so the same colour means the same
# architecture wherever the thesis uses these series.
MODEL_STYLES = {
    "DNN":  {"color": "#1f77b4", "marker": "o"},
    "CNN":  {"color": "#d62728", "marker": "s"},
    "LSTM": {"color": "#2ca02c", "marker": "^"},
    "GRU":  {"color": "#9467bd", "marker": "D"},
}


def _attack_family(name: str) -> str:
    """Map 'fgsm', 'pgd-10', 'pgd-10-fgsminit' -> 'fgsm' or 'pgd'."""
    return "fgsm" if name == "fgsm" else "pgd"


def plot(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(f"{csv_path} is empty")

    # Pull clean accuracy per model (epsilon=0 row).
    clean = df[df["attack"] == "clean"].set_index("model")["accuracy"].to_dict()
    df_attacks = df[df["attack"] != "clean"].copy()
    df_attacks["family"] = df_attacks["attack"].map(_attack_family)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    families = ["fgsm", "pgd"]

    for ax, fam in zip(axes, families):
        sub = df_attacks[df_attacks["family"] == fam]
        for model_name in sub["model"].unique():
            rows = sub[sub["model"] == model_name].sort_values("epsilon")
            style = MODEL_STYLES.get(model_name, {"color": "gray", "marker": "x"})

            # Prepend the clean-accuracy point at eps=0 so each curve starts there.
            xs = [0.0] + rows["epsilon"].tolist()
            ys = [clean.get(model_name, rows["accuracy"].iloc[0])] + rows["accuracy"].tolist()

            ax.plot(xs, ys,
                    label=model_name,
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.8,
                    markersize=6)

        # Random-classifier baseline for 12-class problem (1/12 ≈ 0.083).
        ax.axhline(1 / 12, linestyle=":", color="black", linewidth=0.9, alpha=0.6,
                   label="Random (1/12)")

        ax.set_title(f"{fam.upper()} degradation", fontsize=13)
        ax.set_xlabel(r"$\epsilon$ (L$_\infty$ perturbation budget)")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(left=-0.005)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", fontsize=10, frameon=True)

    axes[0].set_ylabel("Test accuracy")
    fig.suptitle("Adversarial robustness across SEDIKA tier-2 models", fontsize=14, y=1.01)
    fig.tight_layout()

    ensure_dirs(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(OUTPUT_DIR, "adversarial_eval.csv"))
    parser.add_argument("--out", default=os.path.join(PLOT_DIR, "adversarial_degradation.png"))
    args = parser.parse_args()
    plot(args.csv, args.out)


if __name__ == "__main__":
    main()
