"""Quick evaluation of the trained DIFA-2.2 model on held-out target data.

Reports Table 3 numbers: clean accuracy, precision, recall, F1, latency,
plus noisy-domain accuracy at sigma = 0.1.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

from paths import EXTERNAL_DIR, MODEL_DIR, OUTPUT_DIR, ensure_dirs


def main():
    ensure_dirs(OUTPUT_DIR)

    coral_pkl = os.path.join(EXTERNAL_DIR, "sedika_ciciot2023_coral.pkl")
    model_path = os.path.join(MODEL_DIR, "sedika_difa_v2.keras")
    if not os.path.exists(model_path):
        raise SystemExit(f"DIFA model not found: {model_path}")

    df = pd.read_pickle(coral_pkl)
    _train, test = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["target"]
    )
    X = test.drop(columns=["target"]).values.astype(np.float32)
    if X.shape[1] == 25:
        X = np.delete(X, 3, axis=1)
    y = test["target"].values.astype(np.int64)
    print(f"Held-out target test split: {len(X)} samples, {X.shape[1]} features")

    from sedika_difa_v2 import GradientReversal
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"GradientReversal": GradientReversal},
        compile=False,
    )
    print(f"Loaded DIFA model from {model_path}")

    def _predict(X_in):
        task_probs, _domain = model.predict(X_in, batch_size=512, verbose=0)
        return np.argmax(task_probs, axis=1)

    # === Clean evaluation ===
    t0 = time.time()
    y_pred = _predict(X)
    elapsed = time.time() - t0
    n = len(X)
    latency_ms = (elapsed / n) * 1000

    acc_clean = accuracy_score(y, y_pred)
    prec_clean = precision_score(y, y_pred, average="weighted", zero_division=0)
    rec_clean = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1_clean = f1_score(y, y_pred, average="weighted", zero_division=0)
    macro_f1_clean = f1_score(y, y_pred, average="macro", zero_division=0)
    footprint_mb = os.path.getsize(model_path) / (1024 * 1024)

    print("\n=== DIFA-2.2 on held-out target test split (CLEAN) ===")
    print(f"  Accuracy:       {acc_clean:.4f}")
    print(f"  Precision (w):  {prec_clean:.4f}")
    print(f"  Recall (w):     {rec_clean:.4f}")
    print(f"  F1 (weighted):  {f1_clean:.4f}")
    print(f"  F1 (macro):     {macro_f1_clean:.4f}")
    print(f"  Latency:        {latency_ms:.4f} ms/sample")
    print(f"  Footprint:      {footprint_mb:.3f} MB")

    # === Noisy evaluation (sigma = 0.1, matching paper protocol) ===
    rng = np.random.default_rng(42)
    X_noisy = X + rng.normal(0, 0.1, X.shape).astype(np.float32)
    y_pred_n = _predict(X_noisy)
    acc_n = accuracy_score(y, y_pred_n)
    f1_n = f1_score(y, y_pred_n, average="weighted", zero_division=0)
    macro_f1_n = f1_score(y, y_pred_n, average="macro", zero_division=0)
    prec_n = precision_score(y, y_pred_n, average="weighted", zero_division=0)
    rec_n = recall_score(y, y_pred_n, average="weighted", zero_division=0)

    print("\n=== DIFA-2.2 on held-out target test split (NOISY sigma=0.1) ===")
    print(f"  Accuracy:       {acc_n:.4f}")
    print(f"  F1 (weighted):  {f1_n:.4f}")
    print(f"  F1 (macro):     {macro_f1_n:.4f}")

    rows = [
        {
            "scenario": "DIFA (Clean)",
            "accuracy":       round(acc_clean, 4),
            "precision_w":    round(prec_clean, 4),
            "recall_w":       round(rec_clean, 4),
            "f1_weighted":    round(f1_clean, 4),
            "f1_macro":       round(macro_f1_clean, 4),
            "latency_ms":     round(latency_ms, 4),
            "footprint_mb":   round(footprint_mb, 3),
            "n_test_samples": n,
        },
        {
            "scenario": "DIFA (Noisy sigma=0.1)",
            "accuracy":       round(acc_n, 4),
            "precision_w":    round(prec_n, 4),
            "recall_w":       round(rec_n, 4),
            "f1_weighted":    round(f1_n, 4),
            "f1_macro":       round(macro_f1_n, 4),
            "latency_ms":     round(latency_ms, 4),  # same model
            "footprint_mb":   round(footprint_mb, 3),
            "n_test_samples": n,
        },
    ]
    out_csv = os.path.join(OUTPUT_DIR, "difa_target_eval.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
