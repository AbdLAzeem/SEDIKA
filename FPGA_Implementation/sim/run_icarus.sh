#!/usr/bin/env bash
# =====================================================================
# run_icarus.sh
#
# Icarus Verilog (open-source) run script. Useful for quick syntax
# validation when ISE/ModelSim are not available. Verified to compile
# this RTL with Icarus 11+.
#
# Usage:
#   bash run_icarus.sh
#
# Output:
#   ae_tb.vcd  -- VCD waveform for GTKWave
#   stdout     -- PASS/FAIL per vector
# =====================================================================
set -euo pipefail

cd "$(dirname "$0")"

# Copy memory init files alongside the testbench
cp -f ../memory_init/*.mem .

# Compile
iverilog -g2005-sv -o ae_tb.vvp \
    ../rtl/mac_unit.v \
    ../rtl/relu.v \
    ../rtl/weight_rom.v \
    ../rtl/mse_threshold.v \
    ../rtl/ae_top.v \
    ae_tb.v

# Simulate
vvp ae_tb.vvp

echo "Waveform: $(pwd)/ae_tb.vcd  (open with: gtkwave ae_tb.vcd)"
