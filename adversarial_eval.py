"""Adversarial evaluation for SEDIKA models.

Replaces (or complements) Gaussian-jitter robustness with white-box
gradient-based attacks. Reports accuracy degradation curves at multiple
epsilon budgets so reviewers can see boundary deterioration, not just
average-case noise.

Currently supports:
    * FGSM (single-step)        — Goodfellow et al. 2014
    * PGD  (iterative L-inf)    — Madry et al. 2017

Both attacks expect Keras models that output softmax probabilities and
take a flat (samples, features) input. CNN/LSTM/GRU models that need a
3D reshape can be wrapped with `reshape3d=True`.

Usage:
    python adversarial_eval.py
    python adversarial_eval.py --model dnn --epsilons 0.01 0.05 0.1 0.2
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Iterable, List

import numpy as np
import pandas as pd
import tensorflow as tf

from paths import PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR, ensure_dirs


def _to_2d_inputs(X: np.ndarray, reshape3d: bool) -> tf.Tensor:
    """Return a tf.Tensor in the shape the model expects."""
    arr = X.astype(np.float32)
    if reshape3d:
        arr = arr.reshape((arr.shape[0], arr.shape[1], 1))
    return tf.convert_to_tensor(arr)


def _loss(y_true_idx: np.ndarray, y_prob: tf.Tensor) -> tf.Tensor:
    return tf.keras.losses.sparse_categorical_crossentropy(y_true_idx, y_prob)


def fgsm_attack(model: tf.keras.Model, X: np.ndarray, y: np.ndarray,
                epsilon: float, reshape3d: bool = False) -> np.ndarray:
    """Single-step Fast Gradient Sign Method, L-inf budget = epsilon."""
    x_tensor = tf.Variable(_to_2d_inputs(X, reshape3d))
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        preds = model(x_tensor, training=False)
        loss = _loss(y, preds)
    grad = tape.gradient(loss, x_tensor)
    perturbation = epsilon * tf.sign(grad)
    x_adv = x_tensor + perturbation
    return x_adv.numpy().reshape(X.shape)


def pgd_attack(model: tf.keras.Model, X: np.ndarray, y: np.ndarray,
               epsilon: float, alpha: float = None, steps: int = 10,
               reshape3d: bool = False, init: str = "random",
               seed: int = 42) -> np.ndarray:
    """Projected Gradient Descent, L-inf budget = epsilon.

    init:
        "random" (default) — uniform[-eps, eps] perturbation. Standard Madry-style PGD.
        "fgsm"             — start from the FGSM adversarial example. Guarantees
                             PGD effectiveness >= FGSM at the same eps; useful
                             diagnostic when random-init PGD underperforms FGSM
                             due to loss-surface curvature.
        "zero"             — start at the clean input. Pure deterministic PGD.
    """
    if alpha is None:
        alpha = epsilon / 4.0
    x_orig = _to_2d_inputs(X, reshape3d)

    if init == "random":
        tf.random.set_seed(seed)
        x_adv = tf.Variable(x_orig + tf.random.uniform(x_orig.shape, -epsilon, epsilon))
    elif init == "fgsm":
        x_fgsm_2d = fgsm_attack(model, X, y, epsilon=epsilon, reshape3d=reshape3d)
        x_adv = tf.Variable(_to_2d_inputs(x_fgsm_2d, reshape3d))
    elif init == "zero":
        x_adv = tf.Variable(tf.identity(x_orig))
    else:
        raise ValueError(f"unknown pgd init mode: {init!r}")

    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            preds = model(x_adv, training=False)
            loss = _loss(y, preds)
        grad = tape.gradient(loss, x_adv)
        x_adv.assign(x_adv + alpha * tf.sign(grad))
        # Project back into the L-inf ball
        x_adv.assign(tf.clip_by_value(x_adv, x_orig - epsilon, x_orig + epsilon))

    return x_adv.numpy().reshape(X.shape)


def _accuracy(model: tf.keras.Model, X: np.ndarray, y: np.ndarray,
              reshape3d: bool = False) -> float:
    inp = _to_2d_inputs(X, reshape3d)
    preds = model(inp, training=False).numpy()
    return float(np.mean(np.argmax(preds, axis=1) == y))


def evaluate_model(model_name: str, model_path: str, X: np.ndarray, y: np.ndarray,
                   epsilons: Iterable[float], pgd_steps: int = 10,
                   reshape3d: bool = False, pgd_init: str = "random",
                   seed: int = 42) -> List[dict]:
    print(f"\n=== {model_name}  ({model_path})  [pgd_init={pgd_init}] ===")
    model = tf.keras.models.load_model(model_path)
    rows = []

    clean_acc = _accuracy(model, X, y, reshape3d=reshape3d)
    print(f"  clean accuracy: {clean_acc:.4f}")
    rows.append({"model": model_name, "attack": "clean", "epsilon": 0.0,
                 "accuracy": clean_acc, "delta": 0.0})

    pgd_label = f"pgd-{pgd_steps}" if pgd_init == "random" else f"pgd-{pgd_steps}-{pgd_init}init"

    for eps in epsilons:
        x_fgsm = fgsm_attack(model, X, y, epsilon=eps, reshape3d=reshape3d)
        acc_fgsm = _accuracy(model, x_fgsm, y, reshape3d=reshape3d)
        rows.append({"model": model_name, "attack": "fgsm", "epsilon": eps,
                     "accuracy": acc_fgsm, "delta": acc_fgsm - clean_acc})
        print(f"  FGSM eps={eps:.3f}: acc={acc_fgsm:.4f}  (delta={acc_fgsm - clean_acc:+.4f})")

        x_pgd = pgd_attack(model, X, y, epsilon=eps, steps=pgd_steps,
                           reshape3d=reshape3d, init=pgd_init, seed=seed)
        acc_pgd = _accuracy(model, x_pgd, y, reshape3d=reshape3d)
        rows.append({"model": model_name, "attack": pgd_label, "epsilon": eps,
                     "accuracy": acc_pgd, "delta": acc_pgd - clean_acc})
        print(f"  {pgd_label.upper()} eps={eps:.3f}: acc={acc_pgd:.4f}  (delta={acc_pgd - clean_acc:+.4f})")

    return rows


# Each entry: (display_name, filename, needs_3d_reshape)
DEFAULT_MODELS = [
    ("DNN",  "dnn.keras",  False),
    ("CNN",  "cnn.keras",  True),
    ("LSTM", "lstm.keras", True),
    ("GRU",  "gru.keras",  True),
]


def main():
    parser = argparse.ArgumentParser(description="FGSM/PGD adversarial eval for SEDIKA models.")
    parser.add_argument("--model", help="Single model display name to evaluate (default: all available).")
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[0.01, 0.025, 0.05, 0.1, 0.2],
                        help="L-inf perturbation budgets to sweep.")
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--pgd-init", choices=["random", "fgsm", "zero"], default="random",
                        help="PGD initialisation. Use 'fgsm' as a diagnostic when "
                             "random-init PGD underperforms FGSM (loss-surface artefact).")
    parser.add_argument("--sample", type=int, default=2000,
                        help="Random sample size from the test set (eval is grad-tape heavy).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None,
                        help="Override output CSV path (default: results/adversarial_eval.csv).")
    args = parser.parse_args()

    ensure_dirs(OUTPUT_DIR)

    test = pd.read_pickle(os.path.join(PROCESSED_DIR, "test_data.pkl"))
    X = test.drop(columns=["target"]).values.astype(np.float32)
    y = test["target"].values.astype(np.int64)

    if args.sample and args.sample < len(X):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(X), size=args.sample, replace=False)
        X, y = X[idx], y[idx]
        print(f"Sampled {args.sample} test examples (seed={args.seed}).")

    targets = [m for m in DEFAULT_MODELS if (args.model is None or m[0] == args.model)]
    if not targets:
        raise SystemExit(f"Unknown model: {args.model}. Choices: {[m[0] for m in DEFAULT_MODELS]}")

    all_rows = []
    for name, fname, reshape3d in targets:
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f"[skip] {name}: not found at {path}")
            continue
        all_rows.extend(evaluate_model(name, path, X, y, args.epsilons,
                                       pgd_steps=args.pgd_steps, reshape3d=reshape3d,
                                       pgd_init=args.pgd_init, seed=args.seed))

    if not all_rows:
        raise SystemExit("No models evaluated.")

    out_csv = args.out or os.path.join(OUTPUT_DIR, "adversarial_eval.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "attack", "epsilon", "accuracy", "delta"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
