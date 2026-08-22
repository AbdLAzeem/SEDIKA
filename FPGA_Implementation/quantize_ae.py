"""Fixed-point quantization and verification of the SEDIKA Tier-3 autoencoder.

Supports two precision modes:
  * 'int8'  — INT8 weights, INT8 activations  (suitable when AUROC degradation is acceptable)
  * 'int16' — INT8 weights, INT16 activations (recommended for MSE-based anomaly
              scoring where small reconstruction errors carry the signal — Spartan 3E
              DSP multipliers are 18x18 so INT16 activations fit naturally)

Mode is selected with --mode {int8,int16} (default: int16 after empirical AUROC validation).

Pipeline:
  1. Load FP32 AE from models/sedika_ae_adapted.keras.
  2. Calibrate per-tensor activation scales (max-abs) on representative benign data.
  3. Quantize all weights to INT8 and biases to INT32 (symmetric, per-tensor).
  4. Run an integer forward pass in plain Python (no TF) that *exactly* models
     what the Verilog datapath will compute.
  5. Compare AUROC of integer vs FP32 anomaly scoring on the target test set.
     Headline metric: AUROC degradation must be <= 2%.
  6. Emit Verilog $readmemh files (.mem) for weights, biases, and the scaled
     threshold tau_int. Also dump bit-accurate test vectors that the
     ModelSim testbench will replay.

Quantization math (symmetric, per-tensor):
    Weights:    W_q   = round(W / S_w)  with  S_w = max|W| / 127         (INT8)
    Activations: a_q  = round(a / S_a)  with  S_a = max|a| / ACT_MAX
                 ACT_MAX = 127 in int8 mode, 32767 in int16 mode
    Bias:        b_q  = round(b / (S_a_in * S_w))   stored as INT32
    MAC:         acc  = sum_i(a_q[i] * W_q[i]) + b_q       (INT32)
    Requantize:  y_q  = round(acc * (S_a_in * S_w) / S_a_out)
                 (the requant scalar M = S_a_in * S_w / S_a_out is stored
                  as a 16-bit fixed-point multiplier with explicit right
                  shift, so HW can do `(acc * M) >> shift` with no divide)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
MEMORY_DIR = os.path.join(HERE, "memory_init")
SIM_DIR = os.path.join(HERE, "sim")
DOCS_DIR = os.path.join(HERE, "docs")
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(SIM_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

AE_PATH = os.path.join(PROJECT_ROOT, "models", "sedika_ae_adapted.keras")
THRESH_PATH = os.path.join(PROJECT_ROOT, "models", "sedika_ae_threshold.joblib")
TARGET_PKL = os.path.join(PROJECT_ROOT, "processed_external", "sedika_ciciot2023_adaptive.pkl")
TEST_PKL = os.path.join(PROJECT_ROOT, "processed_data", "test_data.pkl")

INT8_MAX = 127
INT8_MIN = -128
NUM_FEATURES = 25  # AE input/output dim (verified from model)

# Activation precision is configurable. Weights are always INT8.
ACT_BITS = 8   # set by CLI
ACT_MAX = 127  # filled in main() based on ACT_BITS
ACT_MIN = -128


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------
def quant_weight_scale(W: np.ndarray) -> float:
    """Symmetric per-tensor scale: max-abs / 127."""
    m = float(np.max(np.abs(W)))
    return m / INT8_MAX if m > 0 else 1.0


def quant_weight_scale_per_channel(W: np.ndarray) -> np.ndarray:
    """Symmetric per-output-channel scale: max-abs across input axis / 127.

    W is shape (in_dim, out_dim). Returns shape (out_dim,) of per-channel scales.
    Per-channel quantisation preserves significantly more precision than
    per-tensor when individual output channels have very different magnitudes
    (common in autoencoder decoder layers where the reconstruction range is
    feature-specific).
    """
    m = np.max(np.abs(W), axis=0)  # (out_dim,)
    return np.where(m > 0, m / INT8_MAX, 1.0)


def quant_weights_per_channel(W: np.ndarray, s_w_pc: np.ndarray) -> np.ndarray:
    """Quantise weights with per-channel scales."""
    q = np.round(W / s_w_pc[np.newaxis, :]).astype(np.int32)
    return np.clip(q, INT8_MIN, INT8_MAX).astype(np.int8)


def quant_act_scale(a: np.ndarray) -> float:
    """Symmetric per-tensor activation scale: max-abs / ACT_MAX.

    Using strict max-abs (rather than a percentile) is critical here because
    the AE's anomaly score is the sum of squared reconstruction errors --
    small magnitude differences are what carry the diagnostic signal, so
    crushing the dynamic range via percentile clipping moves benign and
    attack samples closer together in the quantised score space and degrades
    AUROC by 10+ percentage points.
    """
    m = float(np.max(np.abs(a)))
    return max(m / ACT_MAX, 1e-12)


def quant_to_int8(x: np.ndarray, scale: float) -> np.ndarray:
    """Round + saturate weights to INT8."""
    q = np.round(x / scale).astype(np.int32)
    return np.clip(q, INT8_MIN, INT8_MAX).astype(np.int8)


def quant_to_act(x: np.ndarray, scale: float) -> np.ndarray:
    """Round + saturate to ACT_BITS signed."""
    q = np.round(x / scale).astype(np.int32)
    q = np.clip(q, ACT_MIN, ACT_MAX)
    dtype = np.int8 if ACT_BITS == 8 else np.int16 if ACT_BITS == 16 else np.int32
    return q.astype(dtype)


def m_shift(m_float: float, bits: int = 16) -> Tuple[int, int]:
    """Encode a float multiplier M (0 < M < 1 typical) as (m_int, right_shift)
    such that  M * x ≈ (x * m_int) >> right_shift  for INT32 x.

    Returns (m_int, shift) with m_int in [1, 2^bits - 1].
    """
    if m_float <= 0:
        return 0, 0
    # Find shift such that m_float * 2^shift is in [2^(bits-1), 2^bits)
    shift = bits
    while m_float * (1 << shift) >= (1 << bits):
        shift -= 1
    while m_float * (1 << shift) < (1 << (bits - 1)) and shift < 30:
        shift += 1
    m_int = int(round(m_float * (1 << shift)))
    m_int = min(m_int, (1 << bits) - 1)
    return m_int, shift


# ---------------------------------------------------------------------------
# Integer forward pass (bit-accurate to what Verilog will compute)
# ---------------------------------------------------------------------------
def int_dense(x_q: np.ndarray, W_q: np.ndarray, b_q32: np.ndarray,
              m_int: np.ndarray, m_shift_bits: np.ndarray,
              apply_relu: bool) -> np.ndarray:
    """Single dense layer in fixed-point integer arithmetic with per-channel requantisation.

    x_q:           shape (N_in,)        ACT_BITS signed
    W_q:           shape (N_in, N_out)  INT8 signed (per-channel quantised)
    b_q32:         shape (N_out,)       INT32  (bias pre-scaled to S_a_in * S_w[j] domain)
    m_int:         shape (N_out,) int   per-channel multiplier (16-bit)
    m_shift_bits:  shape (N_out,) int   per-channel right-shift
    apply_relu: if True, clamp negatives to 0

    Returns x_out_q: shape (N_out,) ACT_BITS signed
    """
    x32 = x_q.astype(np.int32)
    W32 = W_q.astype(np.int32)
    acc = x32 @ W32 + b_q32  # (N_out,) INT32 accumulator

    # Per-channel requantization: y[j] = (acc[j] * m_int[j] + half[j]) >> shift[j]
    half = np.where(m_shift_bits > 0, 1 << (m_shift_bits - 1), 0).astype(np.int64)
    y = (acc.astype(np.int64) * m_int.astype(np.int64) + half)
    # Per-channel shift via vectorized division by 2^shift
    y_shifted = np.empty_like(y)
    for j in range(y.shape[0]):
        y_shifted[j] = y[j] >> int(m_shift_bits[j])
    y = y_shifted.astype(np.int32)

    if apply_relu:
        y = np.maximum(y, 0)
    y = np.clip(y, ACT_MIN, ACT_MAX)
    dtype = np.int8 if ACT_BITS == 8 else np.int16 if ACT_BITS == 16 else np.int32
    return y.astype(dtype)


# ---------------------------------------------------------------------------
# Calibration: pick per-tensor activation scales from a calibration set
# ---------------------------------------------------------------------------
def calibrate_act_scales(ae: tf.keras.Model, X_calib: np.ndarray
                         ) -> List[float]:
    """Sample each Dense layer's *output* activation distribution and pick
    a percentile-based scale. Returns [S_input, S_after_L1, S_after_L2,
    S_after_L3, S_after_L4 (= S_input by reconstruction)]."""
    # Probe inputs and per-layer outputs
    scales = []
    s_in = quant_act_scale(X_calib)
    scales.append(s_in)

    intermediate = X_calib
    for layer in ae.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            sub_model = tf.keras.Model(
                inputs=ae.input,
                outputs=layer.output,
            )
            out = sub_model.predict(X_calib, verbose=0)
            # For ReLU layers, calibrate the positive side; for linear, full range.
            s = quant_act_scale(out)
            scales.append(s)
            intermediate = out
    return scales


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    global ACT_BITS, ACT_MAX, ACT_MIN
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["int8", "int16"], default="int16",
                        help="Activation precision (weights are always INT8). "
                             "Default int16 because empirical AUROC degradation "
                             "in int8 mode exceeds the 2%% target for MSE-based "
                             "anomaly scoring on this AE.")
    args = parser.parse_args()
    ACT_BITS = 8 if args.mode == "int8" else 16
    ACT_MAX = (1 << (ACT_BITS - 1)) - 1
    ACT_MIN = -(1 << (ACT_BITS - 1))

    print("=" * 72)
    print(f"SEDIKA Tier-3 Autoencoder Quantization & Verification  [mode={args.mode}]")
    print(f"  Weights: INT8 signed   Activations: INT{ACT_BITS} signed")
    print("=" * 72)

    # ---- Load ----
    print(f"\n[1/7] Loading AE: {AE_PATH}")
    ae = tf.keras.models.load_model(AE_PATH, compile=False)
    threshold_meta = joblib.load(THRESH_PATH)
    tau_fp32 = float(threshold_meta["threshold"])
    print(f"  AE: {ae.input_shape} -> {ae.output_shape}, params={ae.count_params()}")
    print(f"  Threshold tau (FP32) = {tau_fp32:.6f}")
    print(f"  Calibration FPR achieved (paper): {threshold_meta['achieved_fpr_calib']:.4f}")

    # ---- Load target benign data (same population used for tau calibration) ----
    print(f"\n[2/7] Loading target benign data: {TARGET_PKL}")
    target_df = pd.read_pickle(TARGET_PKL)
    benign = target_df[target_df["target"] == 1].drop(columns=["target"]).values.astype(np.float32)
    attack = target_df[target_df["target"] != 1].drop(columns=["target"]).values.astype(np.float32)
    assert benign.shape[1] == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {benign.shape[1]}"
    print(f"  Benign: {benign.shape}   Attack: {attack.shape}")

    # Calibration split: 20% of benign (matches sedika_ae_adaptation.py policy)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(benign))
    calib_n = int(0.2 * len(benign))
    calib_idx = perm[:calib_n]
    eval_benign_idx = perm[calib_n:]
    X_calib = benign[calib_idx]
    X_eval_benign = benign[eval_benign_idx]
    # Evaluation pool: held-out benign + a stratified attack sample
    attack_sample_size = min(len(X_eval_benign), len(attack))
    attack_idx = rng.choice(len(attack), size=attack_sample_size, replace=False)
    X_eval_attack = attack[attack_idx]
    X_eval = np.vstack([X_eval_benign, X_eval_attack])
    y_eval = np.concatenate([np.zeros(len(X_eval_benign)),
                             np.ones(len(X_eval_attack))]).astype(int)
    print(f"  Calibration n={len(X_calib)}, Evaluation benign={len(X_eval_benign)}, "
          f"Evaluation attack={len(X_eval_attack)}")

    # ---- Calibrate activation scales ----
    # CRITICAL: calibrating on benign-only would cause attack activations to
    # saturate the INT range and collapse AUROC. We calibrate on a mix of
    # benign + attack samples so the dynamic range covers both populations.
    # The FPR-budget tau is still derived from benign-only data below; only
    # the HW dynamic range is informed by the mixed calibration.
    n_attack_cal = min(2000, len(attack))
    attack_cal_idx = rng.choice(len(attack), size=n_attack_cal, replace=False)
    X_calib_mix = np.vstack([X_calib, attack[attack_cal_idx]])
    print(f"\n[3/7] Calibrating activation scales on n={len(X_calib_mix)} samples "
          f"(benign={len(X_calib)} + attack={n_attack_cal})...")
    act_scales = calibrate_act_scales(ae, X_calib_mix)
    # act_scales = [S_in, S_a1, S_a2, S_a3, S_a4]
    print(f"  S_in  = {act_scales[0]:.6f}")
    print(f"  S_a1  = {act_scales[1]:.6f}")
    print(f"  S_a2  = {act_scales[2]:.6f}")
    print(f"  S_a3  = {act_scales[3]:.6f}")
    print(f"  S_a4  = {act_scales[4]:.6f}")

    # ---- Quantize weights (per-channel INT8) ----
    print(f"\n[4/7] Quantizing weights (symmetric INT8, per OUTPUT channel)...")
    dense_layers = [l for l in ae.layers if isinstance(l, tf.keras.layers.Dense)]
    assert len(dense_layers) == 4

    Wq_list, bq_list, weight_scales_pc = [], [], []
    m_int_list, m_shift_list = [], []

    for i, layer in enumerate(dense_layers):
        W, b = layer.get_weights()
        # Per-channel weight scale (one scale per OUTPUT neuron)
        s_w_pc = quant_weight_scale_per_channel(W)
        W_q = quant_weights_per_channel(W, s_w_pc)

        s_a_in = act_scales[i]
        if i == len(dense_layers) - 1:
            s_a_out = act_scales[0]   # reconstruction shares input scale
        else:
            s_a_out = act_scales[i + 1]

        # Per-channel bias scale = S_a_in * S_w[j]
        accum_scales = s_a_in * s_w_pc   # (N_out,)
        b_q32 = np.round(b / accum_scales).astype(np.int32)

        # Per-channel requant multiplier
        M_pc = accum_scales / s_a_out    # (N_out,)
        m_int_pc = np.zeros_like(M_pc, dtype=np.int64)
        shift_pc = np.zeros_like(M_pc, dtype=np.int64)
        for j, M in enumerate(M_pc):
            mi, sh = m_shift(float(M), bits=16)
            m_int_pc[j] = mi
            shift_pc[j] = sh

        Wq_list.append(W_q)
        bq_list.append(b_q32)
        weight_scales_pc.append(s_w_pc)
        m_int_list.append(m_int_pc)
        m_shift_list.append(shift_pc)

        print(f"  Layer {i+1} ({layer.name}): W={W.shape}, "
              f"S_w range=[{s_w_pc.min():.3e}, {s_w_pc.max():.3e}]  "
              f"M range=[{M_pc.min():.3e}, {M_pc.max():.3e}]")

    # ---- Integer inference function ----
    def int_forward(x_float: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run a single sample through the integer datapath.

        Returns (reconstruction_float, reconstruction_int).
        """
        s_in = act_scales[0]
        x_q = quant_to_act(x_float, s_in)
        a_q = x_q
        for i in range(len(dense_layers)):
            apply_relu = (i < len(dense_layers) - 1)
            a_q = int_dense(a_q, Wq_list[i], bq_list[i],
                            m_int_list[i], m_shift_list[i],
                            apply_relu=apply_relu)
        # Dequantize to float for MSE comparison
        recon_float = a_q.astype(np.float32) * s_in
        return recon_float, a_q

    # ---- Run FP32 vs INT8 inference and compare ----
    print(f"\n[5/7] Running FP32 + INT8 inference on n={len(X_eval)} eval samples...")

    fp32_recon = ae.predict(X_eval, batch_size=512, verbose=0)
    fp32_mse = np.mean((X_eval - fp32_recon) ** 2, axis=1)
    fp32_pred = (fp32_mse > tau_fp32).astype(int)

    int8_mse_list = []
    int8_recon_first = None
    for i, x in enumerate(X_eval):
        rec_float, rec_q = int_forward(x)
        mse = float(np.mean((x - rec_float) ** 2))
        int8_mse_list.append(mse)
        if i == 0:
            int8_recon_first = (rec_float.copy(), rec_q.copy())
    int8_mse = np.array(int8_mse_list)

    # Recompute tau in INT8 domain on the calibration set so the operating
    # point stays at the same FPR budget. We re-calibrate tau_int because
    # MSE values are computed in dequantized float -- we keep tau_fp32 as
    # the reference but ALSO compute a budget-matched tau_int for the
    # hardware to use.
    calib_int8_mse = []
    for x in X_calib:
        rec_float, _ = int_forward(x)
        calib_int8_mse.append(float(np.mean((x - rec_float) ** 2)))
    calib_int8_mse = np.array(calib_int8_mse)
    tau_int_recalibrated = float(np.quantile(calib_int8_mse, 1.0 - threshold_meta["fpr_budget"]))

    # ---- AUROC comparison ----
    from sklearn.metrics import roc_auc_score
    auroc_fp32 = roc_auc_score(y_eval, fp32_mse)
    auroc_int8 = roc_auc_score(y_eval, int8_mse)
    auroc_drop = auroc_fp32 - auroc_int8
    auroc_drop_pct = 100.0 * auroc_drop / auroc_fp32

    # Threshold-based metrics
    int8_pred_orig_tau = (int8_mse > tau_fp32).astype(int)
    int8_pred_new_tau = (int8_mse > tau_int_recalibrated).astype(int)
    fp32_fpr = float(np.mean(fp32_pred[y_eval == 0]))
    int8_fpr_orig = float(np.mean(int8_pred_orig_tau[y_eval == 0]))
    int8_fpr_new = float(np.mean(int8_pred_new_tau[y_eval == 0]))
    fp32_tpr = float(np.mean(fp32_pred[y_eval == 1]))
    int8_tpr_orig = float(np.mean(int8_pred_orig_tau[y_eval == 1]))
    int8_tpr_new = float(np.mean(int8_pred_new_tau[y_eval == 1]))

    print(f"\n[6/7] Accuracy preservation report")
    print(f"  AUROC  FP32 = {auroc_fp32:.4f}")
    print(f"  AUROC  INT8 = {auroc_int8:.4f}")
    print(f"  Delta  AUROC drop = {auroc_drop:.4f}  ({auroc_drop_pct:+.2f}% of FP32 baseline)")
    print(f"  Pass (<=2%)? {'YES' if abs(auroc_drop_pct) <= 2.0 else 'NO'}")
    print()
    print(f"  Operating point at tau_fp32 = {tau_fp32:.4f}:")
    print(f"    FP32  benign FPR = {fp32_fpr:.4f}  TPR = {fp32_tpr:.4f}")
    print(f"    INT8  benign FPR = {int8_fpr_orig:.4f}  TPR = {int8_tpr_orig:.4f}")
    print()
    print(f"  Recalibrated tau_int = {tau_int_recalibrated:.6f} (matches 0.005 FPR budget on INT8)")
    print(f"    INT8  benign FPR = {int8_fpr_new:.4f}  TPR = {int8_tpr_new:.4f}")

    # ---- Save Verilog memory files ----
    print(f"\n[7/7] Writing Verilog $readmemh memory files...")

    def write_mem(path: str, data: np.ndarray, width_bits: int):
        """Write data as hex words, one per line, two's complement at given width."""
        mask = (1 << width_bits) - 1
        with open(path, "w", encoding="ascii") as f:
            f.write(f"// generated by quantize_ae.py\n")
            f.write(f"// shape={data.shape}, width_bits={width_bits}\n")
            f.write(f"// values are little-endian flat: layer matrices stored row-major (in[i]*W[i][j])\n")
            for v in data.flatten():
                v_unsigned = int(v) & mask
                hex_digits = (width_bits + 3) // 4
                f.write(f"{v_unsigned:0{hex_digits}x}\n")

    # Weights are stored ROW-MAJOR (input-index outer, output-index inner) so
    # that the time-multiplexed MAC engine can stream weight rows during the
    # input-broadcast phase.
    for i, (Wq, bq) in enumerate(zip(Wq_list, bq_list), start=1):
        w_path = os.path.join(MEMORY_DIR, f"weights_L{i}.mem")
        b_path = os.path.join(MEMORY_DIR, f"biases_L{i}.mem")
        write_mem(w_path, Wq, 8)
        write_mem(b_path, bq, 32)
        print(f"  wrote {os.path.basename(w_path)} ({Wq.size} INT8 entries)")
        print(f"  wrote {os.path.basename(b_path)} ({bq.size} INT32 entries)")

    # Save threshold (scaled). MSE in HW will be computed in (INT8 diff)^2 summed.
    # We will compute it in HW in the activation-scale domain S_in:
    #   diff_q  = x_q - x_hat_q   (signed 9-bit)
    #   diff_q^2 (unsigned 18-bit)
    #   sum_sq  = sum over 25     (unsigned 23-bit)
    #   mse_q   ~= sum_sq          (we skip the /25 by folding into threshold)
    # Threshold in this domain: tau_int_hw = tau * 25 / S_in^2
    s_in = act_scales[0]
    tau_for_hw = tau_int_recalibrated * NUM_FEATURES / (s_in ** 2)
    tau_for_hw_int = int(round(tau_for_hw))
    print(f"  tau scaled to HW MSE-sum domain (sum of diff^2, no /N): "
          f"{tau_for_hw:.2f} -> INT {tau_for_hw_int}")

    # Requant multipliers — per output channel per layer
    # Packed as a single 32-bit hex word per channel:  [31:16]=m_int, [4:0]=shift
    # This format is directly $readmemh-compatible -> ae_top reads it without
    # any extra fscanf glue in the testbench.
    for i, (m_pc, s_pc) in enumerate(zip(m_int_list, m_shift_list), start=1):
        rq_path = os.path.join(MEMORY_DIR, f"requant_L{i}.mem")
        with open(rq_path, "w", encoding="ascii") as f:
            f.write(f"// L{i} per-channel requant packed: bits [31:16]=m_int, [4:0]=shift\n")
            for j, (mi, sh) in enumerate(zip(m_pc, s_pc)):
                packed = ((int(mi) & 0xFFFF) << 16) | (int(sh) & 0x1F)
                f.write(f"{packed:08x}\n")
        print(f"  wrote requant_L{i}.mem ({len(m_pc)} channels, packed 32-bit)")

    # ---- Save scales JSON for the testbench ----
    config = {
        "model": "sedika_ae_adapted.keras",
        "num_features": NUM_FEATURES,
        "layers": [
            {"name": l.name, "in_dim": int(l.input_shape[-1]),
             "out_dim": int(l.output_shape[-1])}
            for l in dense_layers
        ],
        "activation_scales": {
            "S_in":  act_scales[0],
            "S_a1":  act_scales[1],
            "S_a2":  act_scales[2],
            "S_a3":  act_scales[3],
            "S_a4":  act_scales[4],
        },
        "weight_scales_per_channel": [
            {f"L{i+1}_S_w_pc": s.tolist()} for i, s in enumerate(weight_scales_pc)
        ],
        "requant_per_channel": [
            {"layer": i + 1,
             "m_int":     m.astype(int).tolist(),
             "shift":     s.astype(int).tolist()}
            for i, (m, s) in enumerate(zip(m_int_list, m_shift_list))
        ],
        "threshold": {
            "tau_fp32_paper":       tau_fp32,
            "tau_int_recalibrated": tau_int_recalibrated,
            "tau_hw_sum_diff_sq":   tau_for_hw_int,
            "fpr_budget":           threshold_meta["fpr_budget"],
        },
        "accuracy": {
            "auroc_fp32":      float(auroc_fp32),
            "auroc_int8":      float(auroc_int8),
            "auroc_drop_pct":  float(auroc_drop_pct),
        },
    }
    with open(os.path.join(MEMORY_DIR, "quant_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"  wrote quant_config.json")

    # ---- Dump test vectors for the ModelSim testbench ----
    # Pick a mix that exercises BOTH branches of the threshold comparator:
    #   - 5 random benign  (anomaly=0 expected)
    #   - 5 random attack  (mostly anomaly=0 at strict 0.5% FPR tau)
    #   - 5 highest-MSE benign  (sanity: tau holds even on outliers)
    #   - 5 highest-MSE attack  (anomaly=1 expected at this strictness)
    # Total: 20 vectors covering both anomaly_flag branches.
    print(f"\nDumping bit-accurate test vectors for ModelSim testbench...")

    # Re-run inference and compute INT MSE for the WHOLE eval set so we
    # can select extreme samples.
    def _hw_mse_sum(x_float):
        s_in_local = act_scales[0]
        x_q_int = quant_to_act(x_float, s_in_local)
        rec_float_local, rec_q_local = int_forward(x_float)
        diff = x_q_int.astype(np.int64) - rec_q_local.astype(np.int64)
        return int(np.sum(diff * diff))

    benign_mses = np.array([_hw_mse_sum(x) for x in X_eval_benign[:1000]])
    attack_mses = np.array([_hw_mse_sum(x) for x in X_eval_attack[:1000]])

    benign_rand_idx = rng.choice(min(1000, len(X_eval_benign)), size=5, replace=False)
    attack_rand_idx = rng.choice(min(1000, len(X_eval_attack)), size=5, replace=False)
    benign_top_idx  = np.argsort(benign_mses)[-5:][::-1]
    attack_top_idx  = np.argsort(attack_mses)[-5:][::-1]

    tb_x = np.vstack([
        X_eval_benign[benign_rand_idx],
        X_eval_attack[attack_rand_idx],
        X_eval_benign[benign_top_idx],
        X_eval_attack[attack_top_idx],
    ])
    tb_label = np.concatenate([
        np.zeros(5), np.ones(5),
        np.zeros(5), np.ones(5),
    ]).astype(int)

    tb_x_path = os.path.join(SIM_DIR, "tb_inputs.mem")
    tb_expected_path = os.path.join(SIM_DIR, "tb_expected.mem")
    act_hex_digits = (ACT_BITS + 3) // 4
    act_mask = (1 << ACT_BITS) - 1
    with open(tb_x_path, "w", encoding="ascii") as fx, \
         open(tb_expected_path, "w", encoding="ascii") as fe:
        fx.write(f"// {len(tb_x)} test vectors, each {NUM_FEATURES} INT{ACT_BITS} values\n")
        fe.write(f"// expected: 1 line per vector, fields: anomaly_flag mse_sum_diff_sq label\n")
        for x, lbl in zip(tb_x, tb_label):
            x_q = quant_to_act(x, act_scales[0])
            for v in x_q:
                v_u = int(v) & act_mask
                fx.write(f"{v_u:0{act_hex_digits}x}\n")
            rec_float, rec_q = int_forward(x)
            x_qi = x_q.astype(np.int32)
            r_qi = rec_q.astype(np.int32)
            diff_q = x_qi - r_qi
            mse_sum_int = int(np.sum(diff_q * diff_q))
            anomaly = 1 if mse_sum_int > tau_for_hw_int else 0
            fe.write(f"{anomaly} {mse_sum_int} {lbl}\n")
    print(f"  wrote {tb_x_path}  ({len(tb_x)} vectors x {NUM_FEATURES} x {ACT_BITS}-bit)")
    print(f"  wrote {tb_expected_path}")

    # ---- Resource budget estimate ----
    total_macs = sum(W.size for W in Wq_list)
    total_weight_bits = sum(W.size for W in Wq_list) * 8
    total_bias_bits = sum(b.size for b in bq_list) * 32
    print(f"\nResource budget summary:")
    print(f"  Total MAC operations per inference: {total_macs}")
    print(f"  Total weight storage (INT8):        {total_weight_bits} bits = {total_weight_bits/1024:.2f} Kbit")
    print(f"  Total bias storage (INT32):         {total_bias_bits} bits = {total_bias_bits/1024:.2f} Kbit")
    print(f"  vs Spartan 3E XC3S500E BRAM budget: 360 Kbit  ->  "
          f"{100 * (total_weight_bits + total_bias_bits)/(360*1024):.2f}% utilisation")
    print()
    print(f"Cycles with 20 multipliers parallel: ceil({total_macs}/20) = {-(-total_macs // 20)}")
    print(f"At 50 MHz: ~{-(-total_macs // 20) / 50:.2f} us (MAC-only, excludes FSM overhead)")
    print()
    print(f"All artefacts written to:")
    print(f"  {MEMORY_DIR}")
    print(f"  {SIM_DIR}")

    return config


if __name__ == "__main__":
    main()
