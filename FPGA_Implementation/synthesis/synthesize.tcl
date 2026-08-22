# =====================================================================
# synthesize.tcl
#
# Xilinx ISE 14.7 batch synthesis script for the SEDIKA Tier-3
# autoencoder on the Spartan 3E XC3S500E.
#
# Usage (Windows command line with ISE 14.7 installed):
#   "C:\Xilinx\14.7\ISE_DS\settings64.bat"
#   xtclsh synthesize.tcl
#
# Output:
#   reports/synthesis_report.txt    -- XST output
#   reports/map_report.txt          -- post-map utilisation
#   reports/par_report.txt          -- post-place-and-route timing
#   sedika_ae.bit                   -- programmable bitstream
#
# This script targets the FPGA flow only; for iSim simulation use the
# accompanying sim/run_isim.do script.
# =====================================================================

# ---- Project setup ----
project new sedika_ae
project set family   "Spartan3E"
project set device   "XC3S500E"
project set package  "FG320"
project set speed    "-4"
project set "Top-Level Source Type"     "HDL"
project set top      "ae_top"

# ---- HDL sources ----
xfile add ../rtl/mac_unit.v
xfile add ../rtl/relu.v
xfile add ../rtl/weight_rom.v
xfile add ../rtl/mse_threshold.v
xfile add ../rtl/ae_top.v

# ---- Memory init files: copy into working dir so $readmemh resolves ----
exec cp ../memory_init/weights_L1.mem .
exec cp ../memory_init/weights_L2.mem .
exec cp ../memory_init/weights_L3.mem .
exec cp ../memory_init/weights_L4.mem .
exec cp ../memory_init/biases_L1.mem .
exec cp ../memory_init/biases_L2.mem .
exec cp ../memory_init/biases_L3.mem .
exec cp ../memory_init/biases_L4.mem .
exec cp ../memory_init/requant_L1.mem .
exec cp ../memory_init/requant_L2.mem .
exec cp ../memory_init/requant_L3.mem .
exec cp ../memory_init/requant_L4.mem .

# ---- Constraints ----
xfile add spartan3e.ucf

# ---- Synthesis options ----
project set "Optimization Goal"           "Speed"
project set "Optimization Effort"         "High"
project set "Keep Hierarchy"              "No"
project set "Other XST Command Line Options" "-use_dsp48 yes -opt_mode speed"

# ---- Run synthesis + implementation ----
process run "Synthesize - XST"
process run "Translate"
process run "Map"
process run "Place & Route"
process run "Generate Programming File"

# ---- Save reports for the paper ----
file mkdir reports
file copy -force [glob *.syr]  reports/synthesis_report.txt
file copy -force [glob *.mrp]  reports/map_report.txt
file copy -force [glob *.par]  reports/par_report.txt

puts "==================================================================="
puts "Synthesis complete. Inspect reports/ for resource utilisation and"
puts "timing. The XST .syr file contains LUT/FF/BRAM/MULT18X18 counts;"
puts "the .par file contains the maximum clock frequency."
puts "==================================================================="

project close
