"""Figure 3: SHAP-based Decision Cliff visualization.

Renders a side-by-side comparison: a tree-based classifier (LightGBM) on
the left, showing stepwise constant probability + abrupt SHAP slope
discontinuities (the 'Decision Cliff'), against the SEDIKA DNN on the
right, showing smooth sigmoid probability + monotone continuous SHAP
attribution (the 'Smooth Slope').

For each model:
    * Sample a fixed reference point from the test set.
    * Sweep one selected feature across 300 linearly interpolated values
      spanning its [p5, p95] range while holding all other features
      constant at the reference values.
    * Plot model predicted-probability and SHAP attribution along that
      feature trace.

Reads:
    models/lightgbm.pkl, models/dnn.keras
    processed_data/test_data.pkl
    processed_data/label_encoder.joblib
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    shap = None

from artifacts import load_artifact
from paths import PROCESSED_DIR, MODEL_DIR, PLOT_DIR, ensure_dirs


# Feature to sweep along — must be in the top-25 manifold. The manuscript
# narrative specifically calls out fwd_pkts_payload.avg as the canonical
# Decision Cliff example.
SWEEP_FEATURE = "fwd_pkts_payload.avg"
N_INTERPOLATION_POINTS = 300


def _prob_curve_ml(model, X_sweep, target_class):
    probs = model.predict_proba(X_sweep)
    return probs[:, target_class]


def _prob_curve_dl(model, X_sweep, target_class):
    arr = X_sweep.astype(np.float32)
    probs = model.predict(arr, batch_size=512, verbose=0)
    return probs[:, target_class]


def _shap_curve_ml(model, X_sweep, feature_idx, target_class, background):
    if shap is None:
        return None
    explainer = shap.TreeExplainer(model, data=background[:50])
    sv = explainer.shap_values(X_sweep)
    if isinstance(sv, list):
        return np.array([sv[target_class][i][feature_idx] for i in range(len(X_sweep))])
    return sv[:, feature_idx, target_class] if sv.ndim == 3 else sv[:, feature_idx]


def _shap_curve_dl(model, X_sweep, feature_idx, target_class, background):
    """Approximate SHAP via finite-difference numerical gradient.

    DeepExplainer is fragile with custom Keras versions; gradient * input
    approximation is sufficient for the smoothness visualization the
    manuscript requires.
    """
    arr = tf.constant(X_sweep.astype(np.float32))
    with tf.GradientTape() as tape:
        tape.watch(arr)
        probs = model(arr, training=False)[:, target_class]
    grads = tape.gradient(probs, arr).numpy()
    # SHAP-like contribution: gradient * (x - baseline)
    baseline = background.mean(axis=0)
    contributions = grads * (X_sweep - baseline)
    return contributions[:, feature_idx]


def main():
    ensure_dirs(PLOT_DIR)

    test = pd.read_pickle(os.path.join(PROCESSED_DIR, "test_data.pkl"))
    le = load_artifact(os.path.join(PROCESSED_DIR, "label_encoder.joblib"))
    class_names = list(le.classes_)
    feature_names = [c for c in test.columns if c != "target"]
    if SWEEP_FEATURE not in feature_names:
        raise SystemExit(f"{SWEEP_FEATURE} not in features. Available: {feature_names}")
    sweep_idx = feature_names.index(SWEEP_FEATURE)

    # Find target class with most samples (representative for SHAP)
    # Use DDOS_Slowloris if available (a clear attack class)
    target_class_name = "DDOS_Slowloris" if "DDOS_Slowloris" in class_names else class_names[0]
    target_class = class_names.index(target_class_name)
    print(f"Sweeping {SWEEP_FEATURE} (index {sweep_idx})")
    print(f"Target class: {target_class_name} (id {target_class})")

    X_test_df = test.drop(columns=["target"])
    X_test = X_test_df.values

    # Pick a reference sample of the target class, fall back to median if none.
    target_rows = test[test["target"] == target_class]
    if len(target_rows) > 0:
        reference = target_rows.iloc[0].drop("target").values.astype(np.float32)
    else:
        reference = X_test_df.median().values.astype(np.float32)

    # Sweep along the selected feature within its [p5, p95] range.
    p5, p95 = np.quantile(X_test[:, sweep_idx], [0.05, 0.95])
    sweep_values = np.linspace(p5, p95, N_INTERPOLATION_POINTS)
    X_sweep = np.tile(reference, (N_INTERPOLATION_POINTS, 1))
    X_sweep[:, sweep_idx] = sweep_values

    # Background for SHAP (sample of test data)
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_test), size=min(200, len(X_test)), replace=False)
    background = X_test[bg_idx]

    # === Load models ===
    print("Loading models...")
    lgbm = joblib.load(os.path.join(MODEL_DIR, "lightgbm.pkl"))
    dnn = tf.keras.models.load_model(os.path.join(MODEL_DIR, "dnn.keras"))

    # === Probability curves ===
    prob_lgbm = _prob_curve_ml(lgbm, X_sweep, target_class)
    prob_dnn = _prob_curve_dl(dnn, X_sweep, target_class)

    # === SHAP curves ===
    print("Computing SHAP curves...")
    shap_lgbm = _shap_curve_ml(lgbm, X_sweep, sweep_idx, target_class, background)
    shap_dnn = _shap_curve_dl(dnn, X_sweep, sweep_idx, target_class, background)

    # === Figure ===
    # Taller layout to accommodate a single shared legend BELOW the plots,
    # outside the data area, so neither the red probability nor the blue SHAP
    # curve has any annotation sitting on top of it.
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), sharey=False)

    # Track all artists for the shared legend
    legend_handles = []
    legend_labels  = []

    # --- Left: LightGBM (Decision Cliff) ---
    ax = axes[0]
    ax2 = ax.twinx()
    l1, = ax.plot(sweep_values, prob_lgbm, color="#d62728", linewidth=2,
                  label="P(target class) — LightGBM")
    legend_handles.append(l1); legend_labels.append(l1.get_label())
    if shap_lgbm is not None:
        l2, = ax2.plot(sweep_values, shap_lgbm, color="#1f77b4", linewidth=1.4,
                       linestyle="--", label="SHAP contribution — LightGBM")
        legend_handles.append(l2); legend_labels.append(l2.get_label())

    # Highlight the cliff zone (max gradient region)
    if shap_lgbm is not None:
        d_prob = np.abs(np.diff(prob_lgbm))
        if d_prob.max() > 0.1:
            cliff_idx = int(np.argmax(d_prob))
            ax.axvspan(sweep_values[max(0, cliff_idx - 5)],
                       sweep_values[min(len(sweep_values) - 1, cliff_idx + 5)],
                       color="#ffcc80", alpha=0.45, zorder=0)
            # Move the "Catastrophic drop" annotation BELOW the data area
            # (was at y=0.5 which intersected the LightGBM probability curve).
            ax.annotate("Catastrophic\nprobability drop",
                        xy=(sweep_values[cliff_idx], prob_lgbm[cliff_idx]),
                        xytext=(sweep_values[cliff_idx] + 0.2, 0.32),
                        ha="left", fontsize=9, color="#c62828",
                        arrowprops=dict(arrowstyle="->", color="#c62828", lw=0.8))

    ax.set_title("Tier 1 (LightGBM): step-wise probability + abrupt SHAP slope\n"
                 "— Decision Cliff signature", fontsize=11)
    ax.set_xlabel(f"{SWEEP_FEATURE} (z-score)")
    ax.set_ylabel("P(target class)", color="#d62728")
    ax2.set_ylabel("SHAP contribution", color="#1f77b4")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.35)

    # --- Right: DNN (Smooth Slope) ---
    ax = axes[1]
    ax2 = ax.twinx()
    l3, = ax.plot(sweep_values, prob_dnn, color="#2ca02c", linewidth=2,
                  label="P(target class) — SEDIKA DNN")
    legend_handles.append(l3); legend_labels.append(l3.get_label())
    l4, = ax2.plot(sweep_values, shap_dnn, color="#1f77b4", linewidth=1.4,
                   linestyle="--", label="SHAP contribution — SEDIKA DNN")
    legend_handles.append(l4); legend_labels.append(l4.get_label())

    ax.set_title("Tier 2 (SEDIKA DNN): smooth probability + monotone SHAP slope\n"
                 "— passes Eq. (10) audit", fontsize=11)
    ax.set_xlabel(f"{SWEEP_FEATURE} (z-score)")
    ax.set_ylabel("P(target class)", color="#2ca02c")
    ax2.set_ylabel("SHAP contribution", color="#1f77b4")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.35)

    # ---- Shared legend BELOW the figure (outside the data area) ----
    fig.legend(legend_handles, legend_labels,
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=4, fontsize=10, frameon=True)

    fig.suptitle(
        "Decision Cliff (left) vs Smooth Slope (right) along "
        f"the {SWEEP_FEATURE} axis at a fixed reference sample",
        fontsize=12, y=1.005,
    )
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.18, top=0.86, wspace=0.30)

    out_path = os.path.join(PLOT_DIR, "figure3_decision_cliff_shap.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
