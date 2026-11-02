# Non-project batch-mode Vivado flow for the Lyra zone_isolation SECDED core
# (Project-10 PRD Phase 2b). No board I/O pins are assigned -- this run is
# purely for resource (LUT/FF) and timing (Fmax) characterisation, not a
# flashable bitstream; ILA/VIO-based real-hardware fault injection (PRD 9.6
# item 2) is a separate follow-up once these numbers are in hand.
#
# Same parenthesized-path workaround as Project-08 (see its CLAUDE.md /
# scripts/vivado_flow.ps1): vivado.bat must be launched from a paren-free cwd
# with paren-free argument paths (D:\lyra_fpga junction -> this project root).
# This script then cd's back into the real project root, which Tcl's `cd`
# tolerates fine.
#
# Usage: scripts/vivado_flow.ps1 -Freq 100 -Part xc7s50csga324-1

set ROOT [file dirname [file dirname [file normalize [info script]]]]
cd $ROOT

set FREQ  [lindex $argv 0]
set PART  [lindex $argv 1]

set tag lyra_zone_isolation_secded
set period [expr {1000.0 / double($FREQ)}]

file mkdir build/vivado reports/vivado

read_verilog -sv [list \
    rtl/secded_decoder.v rtl/lyra_zone_isolation_core.v rtl/lyra_zone_isolation_top.v]

synth_design -top lyra_zone_isolation_top -part $PART
create_clock -name sys_clk -period $period [get_ports clk] -add

write_checkpoint -force build/vivado/${tag}_synth.dcp
report_utilization -file reports/vivado/${tag}_util_synth.txt

opt_design
place_design
phys_opt_design
route_design

write_checkpoint -force build/vivado/${tag}_routed.dcp
report_timing_summary -delay_type min_max -report_unconstrained -max_paths 10 -file reports/vivado/${tag}_timing.txt
report_utilization -file reports/vivado/${tag}_util.txt

puts "DONE tag=$tag part=$PART freq_mhz=$FREQ"
