# SEDIKA Tier-3 Autoencoder — FPGA Resource & Timing Budget

**Target device:** Xilinx Spartan 3E XC3S500E-FG320 (Digilent Spartan 3E Starter Kit, Rev. C)
**Clock:** 50 MHz on-board oscillator (20 ns period)
**Quantization mode:** INT8 weights (per output channel) + INT16 activations
**Bit-accuracy verified:** 20/20 test vectors match Python INT16 reference

---

## 1. Headline numbers for the paper

| Quantity | Value | Method of measurement |
|---|---|---|
| **AUROC degradation (INT8w + INT16a vs FP32)** | **1.21%** | `quantize_ae.py` + `rtl_emulator.py` |
| **FP32 AUROC baseline** | 0.8778 | scikit-learn `roc_auc_score` on n=44,336 mixed eval set |
| **Quantized AUROC** | 0.8672 | same eval set, after INT16 forward pass |
| **MAC operations / inference** | 1,056 | 25·16 + 16·8 + 8·16 + 16·25 |
| **Total weight storage** | 8.25 Kbit (≈ 1.03 KB) | 1,056 INT8 weights |
| **Total bias storage** | 2.03 Kbit | 65 INT32 biases |
| **Active threshold (FPR-budget τ scaled to HW)** | 3,784,122 | `tau_fp32 · N / S_in² = 3.78M` |
| **Estimated MAC-only latency at 50 MHz** | 1.06 μs | 53 cycles · 20 ns (with 20-way parallel MAC) |
| **Estimated total inference latency at 50 MHz** | ≈ 3 – 5 μs | MAC + per-layer requant + MSE (~150 – 250 cycles total) |

The MAC-only latency assumes the FSM schedule documented in `rtl/ae_top.v`:
- Layer 1 (25→16): 25 cycles, 16 active MACs (out of 20)
- Layer 2 (16→8):  16 cycles,  8 active MACs
- Layer 3 (8→16):   8 cycles, 16 active MACs
- Layer 4a (16→20): 16 cycles, 20 MACs (first 20 output channels)
- Layer 4b (16→5):  16 cycles,  5 MACs (remaining 5 output channels)
- **MAC cycles total = 81**
- Plus per-layer finalize (1 cycle each = 5 cycles)
- Plus MSE accumulation (25 cycles)
- **Total cycles ≈ 81 + 5 + 25 = 111**, so ≈ **2.22 μs** end-to-end at 50 MHz.

(The 3 – 5 μs upper bound accounts for FSM hand-off cycles between states.)

---

## 2. Predicted device utilisation on XC3S500E

**These are pre-synthesis estimates derived from operation counts and standard Spartan-3E mapping heuristics. Actual numbers must be captured from the XST `.syr` and Map `.mrp` reports after running `synthesize.tcl`.**

| Resource | XC3S500E budget | Predicted usage | % of budget |
|---|---|---|---|
| 4-input LUTs | 9,312 | ≈ 1,800 – 2,400 | 19 – 26 % |
| Slice flip-flops | 9,312 | ≈ 900 – 1,300 | 10 – 14 % |
| Slices (combined) | 4,656 | ≈ 1,200 – 1,800 | 26 – 39 % |
| BlockRAM (16 Kbit each) | 20 | ≈ 4 – 8 | 20 – 40 % |
| 18×18 multipliers | 20 | **20** (saturated by design) | **100 %** |
| Max f_clk (post-PAR estimate) | — | ≥ 50 MHz | meets 20 ns period |

**Estimate basis:**

- 20 MAC units × ≈ 60 slices/MAC (multiplier + 40-bit accumulator) = ~1,200 slices for MAC array
- Weight/bias storage: 8.25 Kbit + 2.03 Kbit = 10.3 Kbit → fits in 1 BlockRAM (16 Kbit); requantisation tables add 4 small BlockRAMs (one per layer); total ~5–8 BlockRAMs depending on synthesis choice between BRAM and distributed-RAM mapping
- 25-element activation buffers (2 × 25 × 16 bits = 800 bits) → ~50 LUTs each as distributed RAM
- FSM + control = ~200–400 LUTs
- Requantize() function inlined → ~25 slices per output channel × ≤25 channels = ~500 slices in the worst-case single-cycle requant; sharable across cycles if synthesis re-uses

---

## 3. Power estimate (board-level, pre-deployment)

| Component | Estimated mA @ 5 V |
|---|---|
| Spartan 3E XC3S500E core (1.2 V) at 50 MHz, ~30% LUT utilisation | ≈ 80 – 120 mA at 5 V (after on-board regulator) |
| Configuration FLASH + JTAG idle | ≈ 30 mA |
| Board peripherals (LEDs, USB, oscillator) | ≈ 50 – 80 mA |
| **Total board active draw (estimate)** | **≈ 160 – 230 mA** |
| Board idle (FPGA halted) | ≈ 90 – 130 mA |
| **Active vs idle delta (the published "power" figure)** | **≈ 70 – 100 mA × 5 V = 350 – 500 mW** |

**These are projections.** A real measurement should be taken with an inline ammeter on the 5 V power jack of the Starter Kit during a continuous 10,000-inference loop. The Xilinx XPower (or XPA in ISE 14.7) estimator can also produce a closer simulation-based estimate from a `.ncd` + `.pcf` once place-and-route completes.

---

## 4. Comparison framework for the paper

The following table template should be populated after synthesis and used for Section 6 of the FPGA paper:

| Reference | Platform | Target | Latency | Power | Footprint | Accuracy |
|---|---|---|---|---|---|---|
| Proposed (this work) | Spartan 3E XC3S500E | AE-IDS | **[MEASURED]** μs | **[MEASURED]** mW | 0.169 MB (float) / 1.03 KB (INT8 weights) | **AUROC 0.867 (INT16); 1.21% drop vs FP32 0.878** |
| [TBD reference 1] | Zynq-7020 / Artix-7 | DNN-IDS | … | … | … | … |
| [TBD reference 2] | Cyclone V / ECP5 | CNN-IDS | … | … | … | … |
| [TBD reference 3] | ASIC / RISC-V softcore | one-class IDS | … | … | … | … |

Recommended search terms for the related-work column: "FPGA IDS autoencoder", "lightweight anomaly detection FPGA", "INT8 quantization IoT FPGA".

---

## 5. Risk register (what could change these numbers when you synthesize)

1. **Multipliers may exceed 20** if the synthesis tool maps the per-channel requantisation `acc × m` operation to an additional 18×18 multiplier. Mitigation: force `(* USE_DSP48="no" *)` on the requantize step so it builds out of LUTs instead, or pipeline the requant so the multiplier is shared with the MAC array.
2. **BlockRAM count may be 0** if XST chooses to map every storage element to distributed RAM (LUTs). For this design that is acceptable because total weight storage is so small (~10 Kbit). Distributed-RAM mapping will *increase* the LUT count by ≈ 600 LUTs.
3. **f_max may drop below 50 MHz** if the per-channel requantize multiplier creates a long combinational path (16×40 multiplication + variable-shift + saturation in one cycle). Mitigation: register the multiplier output (insert a pipeline stage in `requantize`), which extends per-layer finalize from 1 cycle to 2 cycles (+5 cycles total, ~+100 ns latency).
4. **MEM file initialisation may not synthesize** on older XST versions if relative paths are used. The `synthesize.tcl` already copies `.mem` files into the working directory specifically to avoid this.

---

## 6. Validation chain (paper-ready)

```
FP32 Keras (sedika_ae_adapted.keras)
        |    AUROC = 0.8778
        v
quantize_ae.py  (INT8 weights per-channel + INT16 acts)
        |    AUROC = 0.8672  (1.21% drop, <2% target PASS)
        v
.mem files (weights_Lx, biases_Lx, requant_Lx)
        |
        v
rtl_emulator.py  (bit-accurate Python mirror of ae_top.v)
        |    20/20 vectors match Python INT16 reference  PASS
        v
ae_top.v + sub-modules  (Verilog-2001, ISE 14.7-compatible)
        |    [pending] ModelSim/iSim/Icarus run reproduces emulator
        v
XST + Map + PAR (Xilinx ISE 14.7, Spartan 3E XC3S500E)
        |    [pending] reports/synthesis_report.txt
        v
.bit programming file -> Spartan 3E Starter Kit
        |    [pending] on-board measurement: latency via debug-pin
        |              toggle on oscilloscope; power via inline ammeter
        v
Paper Section 4 (Implementation Results)
```
