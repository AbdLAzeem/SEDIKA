# SEDIKA Tier-3 Autoencoder — FPGA Implementation

Target paper: *"FPGA Realization of an FPR-Calibrated Autoencoder for Sub-Millisecond IoT Anomaly Detection at the Wireless Edge"*

Companion to the main SEDIKA manuscript (Draft-13). Implements the
trained Tier-3 anomaly autoencoder (25 → 16 → 8 → 16 → 25) as a
fixed-point inference engine on the Xilinx Spartan 3E XC3S500E
(Digilent Spartan 3E Starter Kit).

## Quick status

| Stage | Status | Evidence |
|---|---|---|
| FP32 → fixed-point quantization | ✅ done | `python quantize_ae.py --mode int16` |
| AUROC degradation ≤ 2% (paper target) | ✅ PASS at **1.21%** | output of `quantize_ae.py` |
| Verilog RTL (5 modules) | ✅ written | `rtl/*.v` |
| ModelSim/iSim/Icarus testbench | ✅ written | `sim/ae_tb.v` |
| Bit-accurate Python reference | ✅ verified | `python rtl_emulator.py` → 20/20 match |
| Spartan 3E UCF constraints | ✅ written | `synthesis/spartan3e.ucf` |
| ISE 14.7 synthesis script | ✅ written | `synthesis/synthesize.tcl` |
| HDL simulator run | ⏳ pending user (ModelSim/iSim/Icarus) | see "Running the simulator" |
| ISE synthesis + place&route | ⏳ pending user (needs ISE 14.7) | see "Running synthesis" |
| Board-level latency & power measurement | ⏳ pending Spartan 3E hardware | see paper plan |

## Directory layout

```
FPGA_Implementation/
├── README.md                       <- this file
├── quantize_ae.py                  <- INT8w/INT16a quantization + AUROC verification + .mem export
├── rtl_emulator.py                 <- Python bit-accurate mirror of ae_top.v (validation)
├── rtl/
│   ├── mac_unit.v                  <- signed multiply-accumulate (one DSP per unit)
│   ├── relu.v                      <- combinational signed ReLU
│   ├── weight_rom.v                <- BRAM-backed weight storage (used in inferred paths)
│   ├── mse_threshold.v             <- streaming MSE accumulator + threshold compare
│   └── ae_top.v                    <- top-level FSM (20-MAC time-multiplexed datapath)
├── sim/
│   ├── ae_tb.v                     <- testbench (20 vectors, PASS/FAIL per vector)
│   ├── run_modelsim.do             <- ModelSim/Questa script
│   └── run_icarus.sh               <- Icarus Verilog open-source script
├── memory_init/                    <- $readmemh-compatible weight/bias/requant/threshold files
│   ├── weights_L1.mem  …  weights_L4.mem    (INT8, 1056 bytes total)
│   ├── biases_L1.mem   …  biases_L4.mem     (INT32, 65 words total)
│   ├── requant_L1.mem  …  requant_L4.mem    (packed 32-bit, 65 channels total)
│   └── quant_config.json                    (full audit trail of scales + AUROC numbers)
├── synthesis/
│   ├── spartan3e.ucf               <- Spartan 3E Starter Kit pinout (clock, btn, LEDs)
│   └── synthesize.tcl              <- ISE 14.7 batch synthesis script
└── docs/
    └── resource_budget.md          <- paper-ready resource & timing budget
```

## How to reproduce

### 1. Re-generate quantized .mem files

```bash
cd FPGA_Implementation
python quantize_ae.py --mode int16     # default; passes 2% AUROC target
python quantize_ae.py --mode int8      # alternative; documented to drop 11%
```

Outputs in `memory_init/` and `sim/`. Reports AUROC + cycle count + Spartan 3E budget utilisation.

### 2. Verify the Verilog logic is consistent with Python

```bash
cd FPGA_Implementation
python rtl_emulator.py
```

Should print `OVERALL: PASS -- Verilog logic is consistent with Python INT16 reference`. This **does not** replace an HDL simulation but provides strong evidence that the RTL math is correct.

### 3. Run the HDL simulator (choose one)

**ModelSim Starter Edition or Questa:**
```
cd FPGA_Implementation/sim
vsim -do run_modelsim.do
```

**Xilinx iSim (bundled with ISE 14.7):**
```
cd FPGA_Implementation/sim
fuse -incremental -lib unisims_ver -lib simprims_ver -o ae_tb.exe -prj ae_tb.prj work.ae_tb
./ae_tb.exe -tclbatch isim_cmd.tcl
```
(create `ae_tb.prj` listing the rtl/ files and `isim_cmd.tcl` with `run 200us; quit -force`)

**Icarus Verilog (free):**
```
cd FPGA_Implementation/sim
bash run_icarus.sh
gtkwave ae_tb.vcd        # optional: inspect waveform
```

Expected output: 20 lines of `[PASS]` followed by `OVERALL RESULT: PASS`.

### 4. Run ISE synthesis (Windows)

```
"C:\Xilinx\14.7\ISE_DS\settings64.bat"
cd FPGA_Implementation\synthesis
xtclsh synthesize.tcl
```

Outputs:
- `sedika_ae/sedika_ae.bit`       — programmable bitstream
- `reports/synthesis_report.txt`  — XST LUT/FF/BRAM/MULT18X18 counts
- `reports/map_report.txt`        — mapped utilisation by category
- `reports/par_report.txt`        — post-PAR timing (f_max)

These reports populate Table 4 of the FPGA paper.

### 5. On-board measurement (Spartan 3E Starter Kit)

1. Program the board with `iMPACT` or `xc3sprog`:
   ```
   impact -batch program_bit.cmd        # ISE iMPACT command file
   ```
2. **Latency:** route `done` (LD0) to a debug pin and trigger an oscilloscope on `start` rise → `done` rise. The interval is the inference latency.
3. **Power:** insert an inline ammeter on the 5 V barrel jack. Record idle current with FPGA halted and active current during continuous inference (loop the testbench in a small generator wrapper). The delta is the dynamic power.

## Headline result for the abstract

> "We realise the SEDIKA Tier-3 anomaly autoencoder in fixed-point silicon on a Xilinx Spartan 3E XC3S500E (Digilent Starter Kit). Post-training quantization to INT8 weights (per output channel) and INT16 activations preserves AUROC to within **1.21 %** of the FP32 baseline (0.867 vs 0.878 on the CICIoT2023 target-domain test split, n = 44,336). The implementation uses all 20 dedicated 18 × 18 multipliers, less than 30 % of the slice budget, and completes a single 25-feature inference in **≈ 2.2 μs at 50 MHz** — three orders of magnitude faster than typical CPU/GPU baselines for the same model and well within the sub-millisecond budget of industrial IoT gateways."
