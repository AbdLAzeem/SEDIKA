"""Figure 1: SEDIKA Tiered Architecture and Toggle Mechanism.

Renders a simple, publication-quality block diagram of the three-tier
detection pipeline as described in manuscript Section 3.1. Uses
matplotlib (not draw.io) so the figure is reproducible from version
control. For a more polished version, the user can re-create this in
draw.io / TikZ later.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

from paths import PLOT_DIR, ensure_dirs


def _box(ax, xy, w, h, text, fill, edge="#222", text_color="white", fontsize=10, lw=1.4):
    rect = mpatches.FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=lw, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
            ha="center", va="center",
            color=text_color, fontsize=fontsize, weight="bold")


def _arrow(ax, src, dst, color="#444", style="->", lw=1.4):
    arr = FancyArrowPatch(src, dst,
                          arrowstyle=style,
                          mutation_scale=15,
                          color=color, linewidth=lw)
    ax.add_patch(arr)


def main():
    ensure_dirs(PLOT_DIR)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # === Top row: Input + channel monitor ============================
    _box(ax, (0.3, 5.5), 2.4, 1.0, "Network flow\n(packet features)",
         fill="#37474f", fontsize=10)
    _arrow(ax, (2.7, 6.0), (3.4, 6.0))
    _box(ax, (3.4, 5.5), 2.6, 1.0,
         "Channel-condition\nestimator  σ̂",
         fill="#546e7a", fontsize=10)

    # σ branch arrows
    _arrow(ax, (5.0, 5.5), (5.0, 4.4))    # down to router
    ax.text(5.05, 4.95, "σ̂", fontsize=11, weight="bold", color="#37474f")

    # === Middle row: Tier router + three tiers =======================
    _box(ax, (4.0, 3.4), 2.0, 1.0,
         "Tier router",
         fill="#1b5e20", fontsize=10)

    # Tier 1 — Sprint
    _box(ax, (0.3, 1.9), 2.6, 1.2,
         "Tier 1: Sprint\nLightGBM / XGBoost\nσ̂ < 0.01",
         fill="#e64a19", fontsize=9)
    _arrow(ax, (4.0, 3.6), (2.9, 2.5), color="#e64a19", lw=1.6)

    # Tier 2 — Marathon
    _box(ax, (4.0, 1.9), 2.6, 1.2,
         "Tier 2: Marathon\nDNN backbone (DIFA)\n0.01 ≤ σ̂ ≤ 0.10",
         fill="#1565c0", fontsize=9)
    _arrow(ax, (5.0, 3.4), (5.3, 3.1), color="#1565c0", lw=1.6)

    # Tier 3 — Fallback
    _box(ax, (7.7, 1.9), 2.6, 1.2,
         "Tier 3: Fallback\nAutoencoder\nlow-confidence / σ̂ > 0.10",
         fill="#6a1b9a", fontsize=9)
    _arrow(ax, (6.0, 3.6), (7.7, 2.5), color="#6a1b9a", lw=1.6)

    # === Bottom: Threat alert =======================================
    _box(ax, (3.4, 0.3), 5.0, 1.0,
         "Threat alert  ⟶  SHAP audit  ⟶  SOC dashboard",
         fill="#222", fontsize=10)
    _arrow(ax, (1.6, 1.9), (4.5, 1.3), color="#e64a19", lw=1.4)
    _arrow(ax, (5.3, 1.9), (5.3, 1.3), color="#1565c0", lw=1.4)
    _arrow(ax, (9.0, 1.9), (6.8, 1.3), color="#6a1b9a", lw=1.4)

    # === DIFA training-only sidecar ==================================
    _box(ax, (10.4, 1.9), 2.4, 4.6,
         "Training-only:\n\n• CORAL covariance\n   alignment\n\n"
         "• DANN gradient-\n   reversal training\n\n"
         "• λ warm-up\n   scheduler\n\n"
         "• Entropy\n   minimisation",
         fill="#37474f", fontsize=8.5)
    # Dashed annotation
    ax.annotate("", xy=(10.4, 4.0), xytext=(6.6, 2.5),
                arrowprops=dict(arrowstyle="->", linestyle="--",
                                 color="#444", lw=1.2))
    ax.text(8.7, 3.6, "produces", fontsize=8, color="#444", style="italic")

    # === Title ===
    ax.set_title(
        "SEDIKA tiered detection architecture: σ̂-driven routing across "
        "Sprint, Marathon, and Fallback tiers",
        fontsize=12, pad=10,
    )

    out_path = os.path.join(PLOT_DIR, "figure1_architecture.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
