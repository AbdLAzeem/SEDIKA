"""Figure 5: Radar Behavioural Fingerprints.

Renders an 8-axis radar chart comparing the median normalised feature
profile of ARP_Poisoning attacks against benign Thing_Speak traffic.
This is the figure manuscript Section 5.2 references to anchor the
'geometric volume vs axis-aligned cut' narrative.

Reads:
    processed_data/test_data.pkl   (already z-scored, top-25 features)
    processed_data/label_encoder.joblib   (manifest-wrapped)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from artifacts import load_artifact
from paths import PROCESSED_DIR, PLOT_DIR, ensure_dirs


# Feature axes hand-picked by mean absolute SHAP attribution in the
# manuscript text. If any axis is missing from the feature set we drop
# it and pad to 8 from the highest-variance remaining columns.
PREFERRED_AXES = [
    "fwd_pkts_per_sec",
    "bwd_pkts_per_sec",
    "flow_pkts_payload.avg",
    "fwd_pkts_payload.avg",
    "flow_duration",
    "fwd_header_size_tot",
    "flow_iat.avg",
    "fwd_PSH_flag_count",
]


def _normalise_for_radar(values: np.ndarray) -> np.ndarray:
    """Map z-scored values to [0, 1] for radar display.

    z-score features can be negative; for radar visualisation we map per
    axis to [0, 1] using the union (attack + benign) min/max so both
    polygons stay on the same scale.
    """
    return values  # the per-axis range is handled at plotting time


def main():
    ensure_dirs(PLOT_DIR)

    test = pd.read_pickle(os.path.join(PROCESSED_DIR, "test_data.pkl"))
    le = load_artifact(os.path.join(PROCESSED_DIR, "label_encoder.joblib"))
    class_names = list(le.classes_)
    feature_names = [c for c in test.columns if c != "target"]
    print(f"Class names available: {class_names}")
    print(f"Features available: {feature_names}")

    # Resolve class indices (note: 'ARP_poisioning' is the misspelt class label
    # baked into the source dataset; we preserve the source taxonomy).
    def _find_class(*candidates):
        for cand in candidates:
            for i, c in enumerate(class_names):
                if cand.lower() in c.lower():
                    return i
        return None

    arp_idx = _find_class("arp", "poison")
    benign_idx = _find_class("thing_speak", "thing", "benign", "mqtt_publish")
    if arp_idx is None or benign_idx is None:
        raise SystemExit(
            f"Could not resolve ARP or benign class indices. "
            f"Available: {class_names}"
        )
    print(f"ARP class: {class_names[arp_idx]} (id={arp_idx})")
    print(f"Benign class: {class_names[benign_idx]} (id={benign_idx})")

    # Resolve usable axes
    usable_axes = [f for f in PREFERRED_AXES if f in feature_names]
    if len(usable_axes) < 8:
        extra = [f for f in feature_names if f not in usable_axes]
        # Pick the highest-variance remaining columns
        var_rank = test[extra].var().sort_values(ascending=False).index.tolist()
        for cand in var_rank:
            usable_axes.append(cand)
            if len(usable_axes) == 8:
                break
    usable_axes = usable_axes[:8]
    print(f"Radar axes: {usable_axes}")

    arp_median = test[test["target"] == arp_idx][usable_axes].median().values
    ben_median = test[test["target"] == benign_idx][usable_axes].median().values

    # Min-max normalise per axis using the full test-set distribution (5th-95th
    # percentile) so the polygon shows the class's position within the broader
    # feature distribution, not just the contrast between two points.
    p5 = test[usable_axes].quantile(0.05).values
    p95 = test[usable_axes].quantile(0.95).values
    span = np.where(p95 - p5 == 0, 1.0, p95 - p5)
    arp_norm = np.clip((arp_median - p5) / span, 0, 1)
    ben_norm = np.clip((ben_median - p5) / span, 0, 1)

    # Close the polygon
    angles = np.linspace(0, 2 * np.pi, len(usable_axes), endpoint=False).tolist()
    angles += angles[:1]
    arp_plot = np.concatenate([arp_norm, arp_norm[:1]])
    ben_plot = np.concatenate([ben_norm, ben_norm[:1]])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, arp_plot, color="#d62728", linewidth=2, label=f"{class_names[arp_idx]} (attack)")
    ax.fill(angles, arp_plot, color="#d62728", alpha=0.25)
    ax.plot(angles, ben_plot, color="#2ca02c", linewidth=2, label=f"{class_names[benign_idx]} (normal)")
    ax.fill(angles, ben_plot, color="#2ca02c", alpha=0.25)

    # Cosmetic axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(usable_axes, fontsize=9)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Behavioural fingerprint: ARP poisoning vs benign baseline\n"
        "(per-axis envelope-normalised; median over class samples)",
        pad=20, fontsize=12,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=10)

    out_path = os.path.join(PLOT_DIR, "figure5_radar_fingerprint.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
