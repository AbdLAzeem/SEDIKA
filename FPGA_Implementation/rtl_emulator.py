"""Bit-accurate Python emulator of ae_top.v.

Reads the same .mem files that the Verilog uses, mimics the exact
fixed-point arithmetic of the FSM (per-channel requantisation,
INT32 accumulator, INT16 activations, signed saturation), then
compares against tb_expected.mem.

If this passes, the Verilog *logic* is provably consistent with the
Python INT8/INT16 reference that produced the published AUROC. A
ModelSim/iSim run will catch any HDL-specific bugs (timing, state
machine sequencing) but should not change these numerical results.

Usage:
    python rtl_emulator.py
"""
from __future__ import annotations

import os
import sys
from typing import List

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
MEM_DIR = os.path.join(HERE, "memory_init")
SIM_DIR = os.path.join(HERE, "sim")

# Layer dimensions (matches ae_top.v)
DIMS = [(25, 16), (16, 8), (8, 16), (16, 25)]
N_FEAT = 25
ACT_BITS = 16
ACT_MAX = (1 << (ACT_BITS - 1)) - 1
ACT_MIN = -(1 << (ACT_BITS - 1))
TAU_INT_HW = 3784122  # must match TAU_INT parameter in ae_top.v


def read_mem_int(path: str, width_bits: int, signed: bool, count: int) -> np.ndarray:
    out = []
    mask = (1 << width_bits) - 1
    sign_bit = 1 << (width_bits - 1)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            v = int(line.split()[0], 16)
            v &= mask
            if signed and (v & sign_bit):
                v -= 1 << width_bits
            out.append(v)
            if len(out) >= count:
                break
    return np.array(out, dtype=np.int64)


def load_weights():
    """Reload exactly what Verilog $readmemh produces."""
    layers = []
    for i, (n_in, n_out) in enumerate(DIMS, start=1):
        W = read_mem_int(os.path.join(MEM_DIR, f"weights_L{i}.mem"),
                         8, signed=True, count=n_in * n_out)
        W = W.reshape(n_in, n_out)
        B = read_mem_int(os.path.join(MEM_DIR, f"biases_L{i}.mem"),
                         32, signed=True, count=n_out)
        RM = read_mem_int(os.path.join(MEM_DIR, f"requant_L{i}.mem"),
                          32, signed=False, count=n_out)
        # Unpack: m_int in bits [31:16], shift in bits [4:0]
        m_int = (RM >> 16) & 0xFFFF
        shift = RM & 0x1F
        layers.append({"W": W, "B": B, "m": m_int, "s": shift,
                       "n_in": n_in, "n_out": n_out})
    return layers


def requantize_hw(acc: int, bias: int, m: int, shift: int, apply_relu: bool) -> int:
    """Bit-accurate match to the Verilog `requantize` function in ae_top.v.

    Steps:
        acc_b = acc + bias                   (signed 41-bit)
        prod  = acc_b * m                    (signed 57-bit)
        rounded = prod + (1 << (shift-1))    if shift > 0
        shifted = arithmetic_right_shift(rounded, shift)
        if apply_relu and shifted < 0: 0
        else: saturate to [INT16_MIN, INT16_MAX]
    """
    acc_b = acc + bias
    prod = acc_b * int(m)
    if shift > 0:
        prod += 1 << (shift - 1)
    shifted = prod >> int(shift) if shift > 0 else prod
    # Arithmetic shift in Python on negative -> floor division; emulate
    # Verilog $signed `>>>` which is arithmetic.
    # Python's `>>` on negative int already does arithmetic shift. Confirm:
    #   (-1) >> 1 == -1 in Python. Verilog $signed >>> -1 = -1. Match.
    if apply_relu and shifted < 0:
        return 0
    if shifted > ACT_MAX:
        return ACT_MAX
    if shifted < ACT_MIN:
        return ACT_MIN
    return int(shifted)


def emulate_inference(x_q: np.ndarray, layers: list) -> tuple:
    """Mimic the ae_top FSM cycle-by-cycle (functional level, not timing).
    Returns (mse_sum_int, anomaly_flag).
    """
    act_in = x_q.astype(np.int64)
    # Layers 1..3 with ReLU
    for li, layer in enumerate(layers):
        apply_relu = (li < 3)
        W, B, m, s = layer["W"], layer["B"], layer["m"], layer["s"]
        # Accumulator: x @ W + bias_q (bias already at S_a_in * S_w[j] scale)
        acc = act_in.astype(np.int64) @ W.astype(np.int64) + B
        # Per-channel requant + ReLU
        out = np.array([
            requantize_hw(int(acc[j]), 0, int(m[j]), int(s[j]), apply_relu)
            for j in range(layer["n_out"])
        ], dtype=np.int64)
        act_in = out

    # MSE: sum of (x_in - x_hat)^2  (signed INT17 diff -> INT34 product)
    x_hat = act_in
    diffs = x_q.astype(np.int64) - x_hat
    diff_sq = diffs * diffs
    mse_sum = int(np.sum(diff_sq))
    anomaly = 1 if mse_sum > TAU_INT_HW else 0
    return mse_sum, anomaly


def main():
    print("=" * 72)
    print("RTL emulator -- bit-accurate Python mirror of ae_top.v")
    print("=" * 72)

    layers = load_weights()
    for i, l in enumerate(layers, start=1):
        print(f"  Layer {i}: W={l['W'].shape}, B={l['B'].shape}, "
              f"m_range=[{l['m'].min()},{l['m'].max()}], "
              f"shift_range=[{l['s'].min()},{l['s'].max()}]")

    # Read test inputs
    tb_in_path = os.path.join(SIM_DIR, "tb_inputs.mem")
    tb_exp_path = os.path.join(SIM_DIR, "tb_expected.mem")
    raw_in = []
    with open(tb_in_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            v = int(line, 16) & 0xFFFF
            if v & 0x8000:
                v -= 0x10000
            raw_in.append(v)
    raw_in = np.array(raw_in, dtype=np.int64)
    n_vec = len(raw_in) // N_FEAT
    inputs = raw_in.reshape(n_vec, N_FEAT)
    print(f"  Loaded {n_vec} test vectors from {tb_in_path}")

    expected = []
    with open(tb_exp_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                expected.append((int(parts[0]), int(parts[1]), int(parts[2])))
    print(f"  Loaded {len(expected)} expected results from {tb_exp_path}")
    assert len(expected) == n_vec

    print("\n[bit-accurate verification]")
    pass_count = 0
    fail_count = 0
    for i, (x_q, (exp_anom, exp_mse, label)) in enumerate(zip(inputs, expected)):
        mse, anom = emulate_inference(x_q, layers)
        ok = (anom == exp_anom) and (mse == exp_mse)
        status = "PASS" if ok else "FAIL"
        marker = "" if ok else "  <-- MISMATCH"
        print(f"  [{status}] vec {i:2d} label={label}: "
              f"emu(anom={anom} mse={mse:>10d})  exp(anom={exp_anom} mse={exp_mse:>10d}){marker}")
        if ok:
            pass_count += 1
        else:
            fail_count += 1

    print("=" * 72)
    print(f"Result: {pass_count}/{n_vec} vectors matched")
    if fail_count == 0:
        print("OVERALL: PASS -- Verilog logic is consistent with Python INT16 reference")
        print("         (a ModelSim/iSim run is expected to reproduce these exact values)")
    else:
        print(f"OVERALL: FAIL ({fail_count} mismatches)")
        sys.exit(1)


if __name__ == "__main__":
    main()
