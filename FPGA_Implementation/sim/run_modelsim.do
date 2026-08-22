# =====================================================================
# run_modelsim.do
#
# ModelSim Starter Edition / Questa run script for the AE testbench.
# Free ModelSim Starter Edition 10.6+ has been verified to handle this
# design (1k lines of Verilog-2001, single clock domain).
#
# Usage:
#   vsim -do run_modelsim.do
# or from inside ModelSim:
#   do run_modelsim.do
# =====================================================================

# Working directory should be FPGA_Implementation/sim/

if {[file exists work]} {
    vdel -lib work -all
}
vlib work
vmap work work

# Copy memory init files alongside the testbench so the DUT's
# $readmemh calls (which use bare basenames) resolve correctly.
file copy -force ../memory_init/weights_L1.mem  .
file copy -force ../memory_init/weights_L2.mem  .
file copy -force ../memory_init/weights_L3.mem  .
file copy -force ../memory_init/weights_L4.mem  .
file copy -force ../memory_init/biases_L1.mem   .
file copy -force ../memory_init/biases_L2.mem   .
file copy -force ../memory_init/biases_L3.mem   .
file copy -force ../memory_init/biases_L4.mem   .
file copy -force ../memory_init/requant_L1.mem  .
file copy -force ../memory_init/requant_L2.mem  .
file copy -force ../memory_init/requant_L3.mem  .
file copy -force ../memory_init/requant_L4.mem  .

# Compile sources
vlog -work work ../rtl/mac_unit.v
vlog -work work ../rtl/relu.v
vlog -work work ../rtl/weight_rom.v
vlog -work work ../rtl/mse_threshold.v
vlog -work work ../rtl/ae_top.v
vlog -work work ae_tb.v

# Elaborate and run
vsim -voptargs="+acc" -t 1ns work.ae_tb
add wave -position insertpoint sim:/ae_tb/*
add wave -position insertpoint sim:/ae_tb/dut/state
add wave -position insertpoint sim:/ae_tb/dut/cycle
add wave -position insertpoint sim:/ae_tb/dut/mse_sum
run 200us

# Save waveform for the paper figure
# write format wave -window .main_pane.wave.interior.cs.body.pw.wf wave.wlf
puts "============================================================"
puts "ModelSim run complete -- inspect transcript for PASS/FAIL count"
puts "============================================================"
