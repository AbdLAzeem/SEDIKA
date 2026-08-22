"""Cross-domain adversarial evaluation of the adapted DIFA model.

Re-uses the FGSM/PGD primitives from adversarial_eval.py but targets the
DIFA-2.2 adapted model on the CORAL-aligned CICIoT2023 target domain.

This closes a Q1 reviewer gap: prior adversarial evaluation was only on
source-domain models. The more interesting question for a cross-domain
IDS paper is: does adversarial robustness *survive* domain adaptation?

Usage:
    python adversarial_cross_domain.py
    python adversarial_cross_domain.py --epsilons 0.025 0.05 0.1 0.2

Outputs:
    results/adversarial_cross_domain.csv  (model, attack, epsilon, accuracy, delta)
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import pandas as pd
import tensorflow as tf

import adversarial_eval as adv
from paths import EXTERNAL_DIR, MODEL_DIR, OUTPUT_DIR, ensure_dirs


# The DIFA model outputs [task_logits, domain_logits]. We need a wrapper
# that exposes only task_logits so the existing fgsm_attack / pgd_attack
# primitives (which call model(x) and expect a single softmax tensor) work.
class TaskHeadWrapper(tf.keras.Model):
    """Thin wrapper that exposes only the task-classification head."""
    def __init__(self, difa_model: tf.keras.Model):
        super().__init__()
        self.difa = difa_model

    def call(self, x, training=False):
        task_logits, _domain = self.difa(x, training=training)
        return task_logits


def _load_target_test(test_frac=0.2, seed=42):
    """Hold out a target-domain test split that DIFA's training did not see."""
    coral_path = os.path.join(EXTERNAL_DIR, "sedika_ciciot2023_coral.pkl")
    df = pd.read_pickle(coral_path)
    # Stratified hold-out
    from sklearn.model_selection import train_test_split
    _train, test = train_test_split(df, test_size=test_frac,
                                    random_state=seed, stratify=df["target"])
    X = test.drop(columns=["target"]).values.astype(np.float32)
    # Drop the 'no' feature at index 3, matching what DIFA does internally
    if X.shape[1] == 25:
        X = np.delete(X, 3, axis=1)
    y = test["target"].values.astype(np.int64)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difa-model",
                        default=os.path.join(MODEL_DIR, "sedika_difa_v2.keras"))
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[0.01, 0.025, 0.05, 0.1, 0.2])
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--pgd-init", choices=["random", "fgsm", "zero"],
                        default="random")
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out",
                        default=os.path.join(OUTPUT_DIR, "adversarial_cross_domain.csv"))
    args = parser.parse_args()

    ensure_dirs(OUTPUT_DIR)

    if not os.path.exists(args.difa_model):
        raise SystemExit(f"DIFA model not found: {args.difa_model}.\n"
                         f"Run `python sedika_difa_v2.py` first to produce it.")

    # Load DIFA with the custom GRL layer registered
    from sedika_difa_v2 import GradientReversal
    difa_model = tf.keras.models.load_model(
        args.difa_model,
        custom_objects={"GradientReversal": GradientReversal},
        compile=False,
    )
    print(f"Loaded DIFA model: {args.difa_model}")
    wrapped = TaskHeadWrapper(difa_model)

    X, y = _load_target_test(test_frac=args.test_frac, seed=args.seed)
    print(f"Target test split: {len(X)} samples, {X.shape[1]} features")

    if args.sample and args.sample < len(X):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(X), size=args.sample, replace=False)
        X, y = X[idx], y[idx]
        print(f"Sampled {args.sample} examples (seed={args.seed})")

    rows = adv.evaluate_model(
        "DIFA-2.2", args.difa_model, X, y,
        epsilons=args.epsilons,
        pgd_steps=args.pgd_steps,
        reshape3d=False,
        pgd_init=args.pgd_init,
        seed=args.seed,
    ) if False else None
    # We can't call adv.evaluate_model directly because it loads the model
    # itself without custom_objects. Replicate the loop with the wrapped model.

    rows = []
    clean_acc = float(np.mean(
        np.argmax(wrapped(X, training=False).numpy(), axis=1) == y
    ))
    rows.append({"model": "DIFA-2.2", "attack": "clean", "epsilon": 0.0,
                 "accuracy": round(clean_acc, 4), "delta": 0.0})
    print(f"  clean acc on target test: {clean_acc:.4f}")

    pgd_label = (f"pgd-{args.pgd_steps}"
                 if args.pgd_init == "random"
                 else f"pgd-{args.pgd_steps}-{args.pgd_init}init")
    for eps in args.epsilons:
        x_fgsm = adv.fgsm_attack(wrapped, X, y, epsilon=eps, reshape3d=False)
        acc_fgsm = float(np.mean(
            np.argmax(wrapped(x_fgsm.astype(np.float32), training=False).numpy(),
                      axis=1) == y
        ))
        rows.append({"model": "DIFA-2.2", "attack": "fgsm", "epsilon": eps,
                     "accuracy": round(acc_fgsm, 4),
                     "delta": round(acc_fgsm - clean_acc, 4)})
        print(f"  FGSM eps={eps:.3f}: {acc_fgsm:.4f}  (delta={acc_fgsm - clean_acc:+.4f})")

        x_pgd = adv.pgd_attack(wrapped, X, y, epsilon=eps,
                               steps=args.pgd_steps, reshape3d=False,
                               init=args.pgd_init, seed=args.seed)
        acc_pgd = float(np.mean(
            np.argmax(wrapped(x_pgd.astype(np.float32), training=False).numpy(),
                      axis=1) == y
        ))
        rows.append({"model": "DIFA-2.2", "attack": pgd_label, "epsilon": eps,
                     "accuracy": round(acc_pgd, 4),
                     "delta": round(acc_pgd - clean_acc, 4)})
        print(f"  {pgd_label.upper()} eps={eps:.3f}: {acc_pgd:.4f}  (delta={acc_pgd - clean_acc:+.4f})")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "attack", "epsilon", "accuracy", "delta"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
