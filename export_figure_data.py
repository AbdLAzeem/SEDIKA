"""Export the underlying numerical data of each manuscript figure.

Produces one .xlsx file per figure under exports/. Within each file, every
curve is written as its own sheet (and, where useful, also as a combined
'all_curves_long' sheet for easy re-plotting).

Excludes figure1_architecture.png because it is a structural diagram, not
a data plot.

Outputs:
    exports/figure2_accuracy_trap_data.xlsx
    exports/figure3_decision_cliff_shap_data.xlsx
    exports/figure4_difa_convergence_data.xlsx
    exports/figure5_radar_fingerprint_data.xlsx
    exports/adversarial_degradation_data.xlsx
"""
from __future__ import annotations

import os
import re
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from artifacts import load_artifact
from paths import OUTPUT_DIR, MODEL_DIR, PROCESSED_DIR

EXPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 2: Accuracy Trap (per-model accuracy vs sigma)
# ---------------------------------------------------------------------------
def export_figure2():
    ml = pd.read_csv(os.path.join(OUTPUT_DIR, "ml_robustness.csv"))
    dl = pd.read_csv(os.path.join(OUTPUT_DIR, "dl_robustness.csv"))
    ml_clean = pd.read_csv(os.path.join(OUTPUT_DIR, "ml_performance_metrics.csv"))
    dl_clean = pd.read_csv(os.path.join(OUTPUT_DIR, "dl_performance_metrics.csv"))

    clean = {}
    for _, row in ml_clean.iterrows():
        clean[row["Model"]] = float(row["Accuracy"])
    for _, row in dl_clean.iterrows():
        clean[row["Model"]] = float(row["Accuracy"])

    # Reshape into one row per (model, sigma) with the clean point prepended.
    rows = []
    long_rows = []
    for df in (ml, dl):
        for model_name, sub in df.groupby("Base_Model"):
            sub = sub.sort_values("Noise_Level")
            display = "DNN (SEDIKA)" if model_name == "DNN" else model_name
            clean_acc = clean.get(model_name, sub["Accuracy"].iloc[0])

            curve_rows = [(0.0, clean_acc)]
            for _, r in sub.iterrows():
                curve_rows.append((float(r["Noise_Level"]), float(r["Accuracy"])))

            rows.append({
                "model":              display,
                "clean_accuracy":     clean_acc,
                "acc_sigma_0.01":     sub.loc[sub["Noise_Level"] == 0.01, "Accuracy"].iloc[0] if (sub["Noise_Level"] == 0.01).any() else None,
                "acc_sigma_0.05":     sub.loc[sub["Noise_Level"] == 0.05, "Accuracy"].iloc[0] if (sub["Noise_Level"] == 0.05).any() else None,
                "acc_sigma_0.10":     sub.loc[sub["Noise_Level"] == 0.10, "Accuracy"].iloc[0] if (sub["Noise_Level"] == 0.10).any() else None,
                "acc_sigma_0.20":     sub.loc[sub["Noise_Level"] == 0.20, "Accuracy"].iloc[0] if (sub["Noise_Level"] == 0.20).any() else None,
            })
            for sigma, acc in curve_rows:
                long_rows.append({"model": display, "sigma": sigma, "accuracy": acc})

    wide = pd.DataFrame(rows).set_index("model").round(6)
    long = pd.DataFrame(long_rows).round(6)

    out = os.path.join(EXPORTS_DIR, "figure2_accuracy_trap_data.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        wide.to_excel(writer, sheet_name="wide")
        long.to_excel(writer, sheet_name="all_curves_long", index=False)
        # Per-model sheet (one curve = one sheet)
        for model_name, sub in long.groupby("model"):
            sheet = re.sub(r"[^A-Za-z0-9_]+", "_", model_name)[:31]
            sub[["sigma", "accuracy"]].reset_index(drop=True).to_excel(
                writer, sheet_name=sheet, index=False
            )
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Figure 3: SHAP Decision Cliff (re-compute the sweep curves)
# ---------------------------------------------------------------------------
def export_figure3():
    """Re-runs the same sweep as plot_decision_cliff_shap.py and saves the
    underlying x / probability / SHAP values per model."""
    try:
        import shap
    except ImportError:
        print("  [skip] shap not installed")
        return

    SWEEP_FEATURE = "fwd_pkts_payload.avg"
    N_POINTS = 300

    test = pd.read_pickle(os.path.join(PROCESSED_DIR, "test_data.pkl"))
    le = load_artifact(os.path.join(PROCESSED_DIR, "label_encoder.joblib"))
    class_names = list(le.classes_)
    feature_names = [c for c in test.columns if c != "target"]
    sweep_idx = feature_names.index(SWEEP_FEATURE)

    target_class_name = "DDOS_Slowloris" if "DDOS_Slowloris" in class_names else class_names[0]
    target_class = class_names.index(target_class_name)

    X_test_df = test.drop(columns=["target"])
    X_test = X_test_df.values

    target_rows = test[test["target"] == target_class]
    if len(target_rows) > 0:
        reference = target_rows.iloc[0].drop("target").values.astype(np.float32)
    else:
        reference = X_test_df.median().values.astype(np.float32)

    p5, p95 = np.quantile(X_test[:, sweep_idx], [0.05, 0.95])
    sweep_values = np.linspace(p5, p95, N_POINTS)
    X_sweep = np.tile(reference, (N_POINTS, 1))
    X_sweep[:, sweep_idx] = sweep_values

    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_test), size=min(200, len(X_test)), replace=False)
    background = X_test[bg_idx]

    print("  re-loading models for SHAP sweep...")
    lgbm = joblib.load(os.path.join(MODEL_DIR, "lightgbm.pkl"))
    dnn = tf.keras.models.load_model(os.path.join(MODEL_DIR, "dnn.keras"))

    # === LightGBM probability + SHAP ===
    prob_lgbm = lgbm.predict_proba(X_sweep)[:, target_class]
    explainer = shap.TreeExplainer(lgbm, data=background[:50])
    sv = explainer.shap_values(X_sweep)
    if isinstance(sv, list):
        shap_lgbm = np.array([sv[target_class][i][sweep_idx] for i in range(N_POINTS)])
    else:
        shap_lgbm = (sv[:, sweep_idx, target_class]
                     if sv.ndim == 3 else sv[:, sweep_idx])

    # === DNN probability + SHAP-equivalent (gradient * (x - baseline)) ===
    arr = tf.constant(X_sweep.astype(np.float32))
    with tf.GradientTape() as tape:
        tape.watch(arr)
        probs_tensor = dnn(arr, training=False)
        target_probs = probs_tensor[:, target_class]
    grads = tape.gradient(target_probs, arr).numpy()
    prob_dnn = target_probs.numpy()
    baseline = background.mean(axis=0)
    contributions = grads * (X_sweep - baseline)
    shap_dnn = contributions[:, sweep_idx]

    lgbm_df = pd.DataFrame({
        f"{SWEEP_FEATURE}_zscored": sweep_values,
        f"P({target_class_name})":  prob_lgbm,
        f"SHAP_{SWEEP_FEATURE}":    shap_lgbm,
    })
    dnn_df = pd.DataFrame({
        f"{SWEEP_FEATURE}_zscored":      sweep_values,
        f"P({target_class_name})":       prob_dnn,
        f"gradient_input_attribution":   shap_dnn,
    })

    out = os.path.join(EXPORTS_DIR, "figure3_decision_cliff_shap_data.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        lgbm_df.round(6).to_excel(writer, sheet_name="LightGBM_left_panel", index=False)
        dnn_df.round(6).to_excel(writer, sheet_name="DNN_right_panel", index=False)
        meta = pd.DataFrame([
            {"key": "sweep_feature",   "value": SWEEP_FEATURE},
            {"key": "target_class",    "value": target_class_name},
            {"key": "target_class_id", "value": target_class},
            {"key": "n_points",        "value": N_POINTS},
            {"key": "feature_p5",      "value": float(p5)},
            {"key": "feature_p95",     "value": float(p95)},
            {"key": "shap_background_n", "value": 50},
            {"key": "reference_sample_source", "value": f"first test sample of {target_class_name}"},
        ])
        meta.to_excel(writer, sheet_name="metadata", index=False)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Figure 4: DIFA Convergence (parse the log)
# ---------------------------------------------------------------------------
def export_figure4():
    import glob
    logs = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "_difa_rerun*.log")),
        key=os.path.getmtime, reverse=True,
    )
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
            break
    if not rows:
        print("  [skip] no DIFA log lines matched")
        return
    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    df["ln2_reference"] = np.log(2)

    out = os.path.join(EXPORTS_DIR, "figure4_difa_convergence_data.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_curves", index=False)
        # Individual curves
        df[["epoch", "task_loss"]].to_excel(writer,
            sheet_name="task_loss", index=False)
        df[["epoch", "domain_loss", "ln2_reference"]].to_excel(writer,
            sheet_name="domain_loss", index=False)
        df[["epoch", "entropy_loss"]].to_excel(writer,
            sheet_name="entropy_loss", index=False)
        df[["epoch", "alpha_cls", "lambda_dann", "gamma_entropy"]].to_excel(writer,
            sheet_name="loss_weights_schedule", index=False)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Figure 5: Radar fingerprint (per-axis values for both classes)
# ---------------------------------------------------------------------------
def export_figure5():
    PREFERRED_AXES = [
        "fwd_pkts_per_sec", "bwd_pkts_per_sec", "flow_pkts_payload.avg",
        "fwd_pkts_payload.avg", "flow_duration", "fwd_header_size_tot",
        "flow_iat.avg", "fwd_PSH_flag_count",
    ]
    test = pd.read_pickle(os.path.join(PROCESSED_DIR, "test_data.pkl"))
    le = load_artifact(os.path.join(PROCESSED_DIR, "label_encoder.joblib"))
    class_names = list(le.classes_)
    feature_names = [c for c in test.columns if c != "target"]

    def _find(*cands):
        for c in cands:
            for i, name in enumerate(class_names):
                if c.lower() in name.lower():
                    return i
        return None
    arp_idx = _find("arp", "poison")
    benign_idx = _find("thing_speak", "thing", "benign", "mqtt_publish")

    usable_axes = [f for f in PREFERRED_AXES if f in feature_names]
    if len(usable_axes) < 8:
        extra = [f for f in feature_names if f not in usable_axes]
        var_rank = test[extra].var().sort_values(ascending=False).index.tolist()
        for cand in var_rank:
            usable_axes.append(cand)
            if len(usable_axes) == 8:
                break
    usable_axes = usable_axes[:8]

    arp_median = test[test["target"] == arp_idx][usable_axes].median()
    ben_median = test[test["target"] == benign_idx][usable_axes].median()
    p5 = test[usable_axes].quantile(0.05)
    p95 = test[usable_axes].quantile(0.95)
    span = (p95 - p5).where(p95 - p5 != 0, other=1.0)

    arp_norm = ((arp_median - p5) / span).clip(0, 1)
    ben_norm = ((ben_median - p5) / span).clip(0, 1)

    df = pd.DataFrame({
        "axis":             usable_axes,
        "ARP_median_raw":   arp_median.values,
        "Benign_median_raw": ben_median.values,
        "p5":               p5.values,
        "p95":              p95.values,
        "ARP_normalised":   arp_norm.values,
        "Benign_normalised": ben_norm.values,
    }).round(6)

    out = os.path.join(EXPORTS_DIR, "figure5_radar_fingerprint_data.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_axes", index=False)
        # Separate sheets per class (matching plotted polygons)
        df[["axis", "ARP_normalised", "ARP_median_raw"]].to_excel(
            writer, sheet_name=f"ARP_{class_names[arp_idx]}"[:31], index=False
        )
        df[["axis", "Benign_normalised", "Benign_median_raw"]].to_excel(
            writer, sheet_name=f"Benign_{class_names[benign_idx]}"[:31], index=False
        )
        meta = pd.DataFrame([
            {"key": "ARP_class_name",    "value": class_names[arp_idx]},
            {"key": "ARP_class_id",      "value": arp_idx},
            {"key": "Benign_class_name", "value": class_names[benign_idx]},
            {"key": "Benign_class_id",   "value": benign_idx},
            {"key": "normalisation",     "value": "(value - p5)/(p95 - p5) clipped to [0,1]"},
            {"key": "n_axes",            "value": len(usable_axes)},
        ])
        meta.to_excel(writer, sheet_name="metadata", index=False)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# adversarial_degradation.png: per-model FGSM/PGD curves
# ---------------------------------------------------------------------------
def export_adversarial():
    csv = pd.read_csv(os.path.join(OUTPUT_DIR, "adversarial_eval.csv"))

    # Long form already in CSV; we add a wide pivot per model for convenience.
    out = os.path.join(EXPORTS_DIR, "adversarial_degradation_data.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        csv.round(6).to_excel(writer, sheet_name="all_curves_long", index=False)
        for model in csv["model"].unique():
            sub = csv[csv["model"] == model].copy()
            # Pivot: rows = epsilon, columns = (clean | fgsm | pgd-10)
            pivot = sub.pivot_table(index="epsilon", columns="attack",
                                     values="accuracy").reset_index()
            sheet = re.sub(r"[^A-Za-z0-9_]+", "_", model)[:31]
            pivot.round(6).to_excel(writer, sheet_name=sheet, index=False)

    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("Exporting figure data to Excel files in:", EXPORTS_DIR)
    print()
    print("[1/5] Figure 2 (Accuracy Trap)...")
    export_figure2()
    print("[2/5] Figure 3 (SHAP Decision Cliff) -- re-running SHAP sweep (~2 min)...")
    export_figure3()
    print("[3/5] Figure 4 (DIFA Convergence)...")
    export_figure4()
    print("[4/5] Figure 5 (Radar Fingerprint)...")
    export_figure5()
    print("[5/5] Adversarial Degradation...")
    export_adversarial()
    print()
    print("All exports written to:", EXPORTS_DIR)


if __name__ == "__main__":
    main()
