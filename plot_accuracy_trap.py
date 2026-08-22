"""Figure 2: 'Accuracy Trap' — accuracy under Gaussian noise injection.

Plots accuracy vs sigma for the catastrophic-collapse families
(LightGBM, XGBoost, DecisionTree) against the robust families
(DNN, CNN, LSTM, GRU, SVM, KNN). Shades the 'lab condition' and
'realistic interference' zones to support the manuscript narrative
in Section 4.2.

Reads:
    results/ml_robustness.csv
    results/dl_robustness.csv
    results/ml_performance_metrics.csv  (for clean accuracy)
    results/dl_performance_metrics.csv  (for clean accuracy)
"""
from __future__ import annotations

import os

import pandas as pd
import matplotlib.pyplot as plt

from paths import OUTPUT_DIR, PLOT_DIR, ensure_dirs


# Visual identity. Catastrophic-collapse models get warm colours and dashed lines;
# robust models get cool colours and solid lines. The contrast is the figure's point.
STYLES = {
    "LightGBM":      {"color": "#d62728", "linestyle": "--", "marker": "s"},  # red dashed
    "XGBoost":       {"color": "#ff7f0e", "linestyle": "--", "marker": "v"},  # orange dashed
    "DecisionTree":  {"color": "#8c564b", "linestyle": "--", "marker": "x"},  # brown dashed
    "RandomForest":  {"color": "#e377c2", "linestyle": ":",  "marker": "+"},  # pink dotted (partial cliff)
    "DNN (SEDIKA)":  {"color": "#2ca02c", "linestyle": "-",  "marker": "o"},  # green solid
    "CNN":           {"color": "#1f77b4", "linestyle": "-",  "marker": "^"},  # blue
    "LSTM":          {"color": "#17becf", "linestyle": "-",  "marker": "D"},
    "GRU":           {"color": "#9467bd", "linestyle": "-",  "marker": "P"},
    "SVM":           {"color": "#7f7f7f", "linestyle": "-",  "marker": "h"},
    "KNN":           {"color": "#bcbd22", "linestyle": "-",  "marker": "*"},
}


def _load_clean_accuracy():
    """Build {display_name -> clean_accuracy} from the two performance CSVs."""
    ml = pd.read_csv(os.path.join(OUTPUT_DIR, "ml_performance_metrics.csv"))
    dl = pd.read_csv(os.path.join(OUTPUT_DIR, "dl_performance_metrics.csv"))
    clean = {}
    for _, row in ml.iterrows():
        # Map "Random Forest" -> "RandomForest" etc to match robustness CSV
        key = row["Model"].replace(" ", "")
        # Also keep the original-with-space form because robustness CSVs use the spaced version
        clean[row["Model"]] = float(row["Accuracy"])
        clean[key] = float(row["Accuracy"])
    for _, row in dl.iterrows():
        name = row["Model"]
        clean[name] = float(row["Accuracy"])
        if name == "DNN":
            clean["DNN (SEDIKA)"] = float(row["Accuracy"])
    return clean


def _gather_series():
    """Return dict {display_name -> (sigmas, accuracies)} with clean point prepended."""
    ml = pd.read_csv(os.path.join(OUTPUT_DIR, "ml_robustness.csv"))
    dl = pd.read_csv(os.path.join(OUTPUT_DIR, "dl_robustness.csv"))

    # Robustness CSV uses "Random Forest" with space; harmonise to RandomForest where useful.
    def _canon(name):
        return name.replace(" ", "")
    ml["Base_Model_canon"] = ml["Base_Model"].apply(_canon)
    dl["Base_Model_canon"] = dl["Base_Model"]

    clean = _load_clean_accuracy()

    series = {}
    for df in (ml, dl):
        for canon_name, sub in df.groupby("Base_Model_canon"):
            sub_sorted = sub.sort_values("Noise_Level")
            # Display name: rename DNN -> 'DNN (SEDIKA)' for narrative clarity.
            display = "DNN (SEDIKA)" if canon_name == "DNN" else canon_name
            sigmas = [0.0] + sub_sorted["Noise_Level"].tolist()
            accs   = [clean.get(canon_name, clean.get(display, None))]
            accs  += sub_sorted["Accuracy"].tolist()
            if accs[0] is None:
                accs[0] = accs[1]  # fallback
            series[display] = (sigmas, accs)
    return series


def main():
    ensure_dirs(PLOT_DIR)
    series = _gather_series()

    # Wider figure so the legend can sit OUTSIDE the data area on the right
    # without overlapping any of the 10 model curves.
    fig, ax = plt.subplots(figsize=(13, 6.2))

    # Background zone shading
    ax.axvspan(0, 0.05, color="#cfeac0", alpha=0.35, zorder=0)   # lab
    ax.axvspan(0.05, 0.21, color="#f6cccc", alpha=0.25, zorder=0)  # realistic interference

    # Plot order: robust families first (under the dashed collapse lines visually)
    plot_order = [
        "DNN (SEDIKA)", "CNN", "GRU", "LSTM", "SVM", "KNN",
        "RandomForest", "DecisionTree", "XGBoost", "LightGBM",
    ]
    for name in plot_order:
        if name not in series:
            continue
        xs, ys = series[name]
        style = STYLES.get(name, {"color": "gray", "linestyle": "-", "marker": "o"})
        ax.plot(xs, ys,
                label=name,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=1.8 if style["linestyle"] == "--" else 1.6,
                markersize=6,
                alpha=0.95)

    # Annotations
    ax.set_title(
        "The Accuracy Trap: classifier resilience under Gaussian feature perturbation",
        fontsize=13,
    )
    ax.set_xlabel(r"Perturbation magnitude  $\sigma$  (standard deviations on normalised features)")
    ax.set_ylabel("Test accuracy")
    ax.set_xlim(-0.005, 0.205)
    ax.set_ylim(0, 1.04)
    ax.grid(True, linestyle="--", alpha=0.45)

    # Zone labels moved to the TOP of the plot so they never cross the LightGBM
    # collapse trace (which dips to ~0.10 at sigma=0.2). Background shading
    # already conveys the zone boundary; the labels are just text tags.
    ax.text(0.025, 1.005, "Lab conditions (σ ≤ 0.05)",
            fontsize=9.5, color="#2a662a", ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#e6f5da",
                      edgecolor="#2a662a", linewidth=0.6))
    ax.text(0.13, 1.005, "Realistic interference (σ ≥ 0.10)",
            fontsize=9.5, color="#8a2a2a", ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fbe2e2",
                      edgecolor="#8a2a2a", linewidth=0.6))

    # Legend OUTSIDE the data area on the right.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=10, frameon=True, ncol=1)

    # Adjust layout so the external legend isn't clipped at save time.
    fig.subplots_adjust(left=0.07, right=0.83, bottom=0.10, top=0.92)

    out_path = os.path.join(PLOT_DIR, "figure2_accuracy_trap.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
