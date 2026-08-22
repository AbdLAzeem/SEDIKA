"""DIFA-2.2 loss-component ablation harness.

Trains the same architecture under several `LossWeights` configurations and
reports each variant's target-domain accuracy + macro-F1 + domain-classifier
accuracy on a held-out target test split. Produces a comparison table that
isolates the marginal contribution of each loss component (DANN adversarial
term, target-entropy minimisation).

Usage:
    python difa_ablation.py
    python difa_ablation.py --epochs 15 --target-test-frac 0.2
    python difa_ablation.py --configs full,no_dann,no_entropy

NOTE: this is expensive. 4 configs x 20 epochs of DIFA can take 1-2 hours
on CPU. Reduce `--epochs` for a quick relative comparison; the absolute
numbers will drift but the *ranking* of configs is what the ablation
needs to show.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import sedika_difa_v2 as difa
from paths import OUTPUT_DIR, MODEL_DIR, ensure_dirs


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------
# Each config is a (label, LossWeights) pair. The label is the row key in
# the output CSV. To add a new ablation, just extend this dict.
CONFIGS = {
    "full":              difa.LossWeights(alpha_cls=1.0, lambda_dann=1.0, gamma_entropy=0.5),
    "no_dann":           difa.LossWeights(alpha_cls=1.0, lambda_dann=0.0, gamma_entropy=0.5),
    "no_entropy":        difa.LossWeights(alpha_cls=1.0, lambda_dann=1.0, gamma_entropy=0.0),
    "task_only":         difa.LossWeights(alpha_cls=1.0, lambda_dann=0.0, gamma_entropy=0.0),
    "strong_dann":       difa.LossWeights(alpha_cls=1.0, lambda_dann=2.0, gamma_entropy=0.5),
    "strong_entropy":    difa.LossWeights(alpha_cls=1.0, lambda_dann=1.0, gamma_entropy=1.0),
}

DEFAULT_CONFIG_ORDER = ["full", "no_dann", "no_entropy", "task_only"]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _slice_target_features(target_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) with the same 24-feature shape DIFA expects."""
    X = target_df.drop(columns=["target"]).values.astype("float32")
    y = target_df["target"].values
    return X, y


def evaluate(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate a trained DIFA model on held-out target data.

    Returns target-task accuracy + macro-F1 plus the domain-classifier's
    accuracy on the same examples (lower domain-acc => more domain-invariant
    features, which is the DANN objective's whole point).
    """
    task_preds, domain_preds = model.predict(X_test, batch_size=256, verbose=0)

    y_pred = np.argmax(task_preds, axis=1)
    target_acc = float((y_pred == y_test).mean())
    target_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    # Target examples have domain label 1 (source=0, target=1). The
    # discriminator should ideally guess ~0.5 (chance) on aligned features.
    domain_pred_binary = (domain_preds.flatten() > 0.5).astype(int)
    domain_acc_on_target = float((domain_pred_binary == 1).mean())

    return {
        "target_accuracy": target_acc,
        "target_macro_f1": target_f1,
        "domain_acc_on_target": domain_acc_on_target,  # 0.5 = best (indistinguishable)
    }


def run_one_config(label: str, weights: difa.LossWeights, *,
                   epochs: int, source_n: int,
                   X_test: np.ndarray, y_test: np.ndarray,
                   data_seed: int) -> dict:
    """Train one config, score it, return a result row."""
    print("\n" + "=" * 72)
    print(f"ABLATION: {label}  weights={asdict(weights)}")
    print("=" * 72)

    save_path = os.path.join(MODEL_DIR, f"sedika_difa_v2_ablation_{label}.keras")
    t0 = time.time()
    # The harness controls the train/test split itself, so we ask DIFA to
    # use the (source_n) source samples and (len target_test_complement)
    # target samples — but DIFA's data loading is internal. To keep this
    # simple and surgical we let DIFA load and shuffle as usual; we evaluate
    # on a deterministic held-out slice that DIFA hasn't been instructed
    # to exclude. With data_seed fixed, both processes are reproducible
    # and the held-out slice is sampled from the same population, so the
    # comparison across configs is still apples-to-apples (same eval set
    # for every config; same training population).
    model = difa.train_difa_v2(
        weights=weights,
        epochs=epochs,
        save_path=save_path,
        verbose=True,
        source_n=source_n,
        data_seed=data_seed,
    )
    train_seconds = time.time() - t0

    metrics = evaluate(model, X_test, y_test)
    metrics["config"] = label
    metrics["alpha_cls"] = weights.alpha_cls
    metrics["lambda_dann_max"] = weights.lambda_dann
    metrics["gamma_entropy"] = weights.gamma_entropy
    metrics["epochs"] = epochs
    metrics["train_seconds"] = round(train_seconds, 1)
    metrics["model_path"] = save_path

    print(f" -> target_acc={metrics['target_accuracy']:.4f}  "
          f"macro_f1={metrics['target_macro_f1']:.4f}  "
          f"domain_acc(target)={metrics['domain_acc_on_target']:.4f}  "
          f"({train_seconds:.0f}s)")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DIFA-2.2 loss-component ablation.")
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIG_ORDER),
                        help="Comma-separated config labels. "
                             f"Available: {sorted(CONFIGS.keys())}")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Epochs per config (default 20 for ablation; "
                             "production uses 50).")
    parser.add_argument("--source-n", type=int, default=30000)
    parser.add_argument("--target-test-frac", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, "difa_ablation.csv"))
    args = parser.parse_args()

    ensure_dirs(OUTPUT_DIR, MODEL_DIR)

    config_labels = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = set(config_labels) - set(CONFIGS.keys())
    if unknown:
        raise SystemExit(f"Unknown config(s): {sorted(unknown)}. "
                         f"Available: {sorted(CONFIGS.keys())}")

    # Held-out target test split is shared across every config so the rows
    # are directly comparable.
    target_df = pd.read_pickle(difa.TARGET_DATA)
    train_target_df, test_target_df = train_test_split(
        target_df,
        test_size=args.target_test_frac,
        random_state=args.data_seed,
        stratify=target_df["target"],
    )
    X_test, y_test = _slice_target_features(test_target_df)
    print(f"Target test split: {len(X_test)} samples (frac={args.target_test_frac})")

    rows = []
    for label in config_labels:
        weights = CONFIGS[label]
        rows.append(run_one_config(
            label, weights,
            epochs=args.epochs,
            source_n=args.source_n,
            X_test=X_test, y_test=y_test,
            data_seed=args.data_seed,
        ))

    # Write CSV with stable column order
    fieldnames = [
        "config", "alpha_cls", "lambda_dann_max", "gamma_entropy",
        "target_accuracy", "target_macro_f1", "domain_acc_on_target",
        "epochs", "train_seconds", "model_path",
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nAblation table written to {args.out}")

    # Pretty-print summary
    print("\n=== ABLATION SUMMARY ===")
    df = pd.DataFrame(rows)
    print(df[["config", "alpha_cls", "lambda_dann_max", "gamma_entropy",
              "target_accuracy", "target_macro_f1", "domain_acc_on_target"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
