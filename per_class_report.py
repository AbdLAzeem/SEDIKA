"""Per-class precision/recall/F1 report for all trained ML/DL models.

Outputs:
    results/per_class_report_<model>.csv  -- per-class metrics
    results/per_class_summary.csv          -- macro/weighted summary across models

Use the numbers for the per-class F1 table required by Section 4.4 of the manuscript.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

from artifacts import load_artifact
from paths import PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR, ensure_dirs


ML_MODELS = {
    "RandomForest":  "random_forest.pkl",
    "LightGBM":      "lightgbm.pkl",
    "XGBoost":       "xgboost.pkl",
    "DecisionTree":  "decision_tree.pkl",
    "KNN":           "knn.pkl",
    "SVM":           "svm.pkl",
}
DL_MODELS = {
    "DNN":  ("dnn.keras",  False),
    "CNN":  ("cnn.keras",  True),
    "LSTM": ("lstm.keras", True),
    "GRU":  ("gru.keras",  True),
}


def _ml_predict(model, X):
    return model.predict(X)


def _dl_predict(model, X, reshape3d):
    arr = X.values.astype(np.float32)
    if reshape3d:
        arr = arr.reshape((arr.shape[0], arr.shape[1], 1))
    probs = model.predict(arr, batch_size=512, verbose=0)
    return np.argmax(probs, axis=1)


def main():
    ensure_dirs(OUTPUT_DIR)
    test = pd.read_pickle(os.path.join(PROCESSED_DIR, "test_data.pkl"))
    X_test = test.drop(columns=["target"])
    y_test = test["target"].values

    le = load_artifact(os.path.join(PROCESSED_DIR, "label_encoder.joblib"))
    class_names = [str(c) for c in le.classes_]

    summary = []

    for name, fname in ML_MODELS.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f"[skip] {name}: {path}")
            continue
        model = joblib.load(path)
        y_pred = _ml_predict(model, X_test)
        _emit(name, y_test, y_pred, class_names, summary)

    for name, (fname, reshape3d) in DL_MODELS.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f"[skip] {name}: {path}")
            continue
        model = tf.keras.models.load_model(path)
        y_pred = _dl_predict(model, X_test, reshape3d)
        _emit(name, y_test, y_pred, class_names, summary)

    pd.DataFrame(summary).to_csv(
        os.path.join(OUTPUT_DIR, "per_class_summary.csv"), index=False
    )
    print(f"\nWrote summary: {os.path.join(OUTPUT_DIR, 'per_class_summary.csv')}")


def _emit(name, y_test, y_pred, class_names, summary):
    print(f"\n=== {name} ===")
    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True, zero_division=0,
    )
    rows = []
    for cls, m in report.items():
        if cls in ("accuracy", "macro avg", "weighted avg"):
            continue
        rows.append({
            "class": cls,
            "precision": round(m["precision"], 4),
            "recall":    round(m["recall"], 4),
            "f1":        round(m["f1-score"], 4),
            "support":   int(m["support"]),
        })
    df = pd.DataFrame(rows).sort_values("support", ascending=False)
    out_path = os.path.join(OUTPUT_DIR, f"per_class_report_{name}.csv")
    df.to_csv(out_path, index=False)

    macro = report["macro avg"]
    weighted = report["weighted avg"]
    summary.append({
        "model": name,
        "accuracy":            round(report["accuracy"], 4),
        "macro_precision":     round(macro["precision"], 4),
        "macro_recall":        round(macro["recall"], 4),
        "macro_f1":            round(macro["f1-score"], 4),
        "weighted_f1":         round(weighted["f1-score"], 4),
        "min_per_class_f1":    round(df["f1"].min(), 4),
        "max_per_class_f1":    round(df["f1"].max(), 4),
        "n_classes_below_0.5": int((df["f1"] < 0.5).sum()),
    })
    print(f"  acc={report['accuracy']:.4f} macro_f1={macro['f1-score']:.4f}")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
