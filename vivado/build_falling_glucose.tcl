# Non-project batch Vivado flow for the SECOND hardware-validated constraint
# (Medical/falling_glucose_guard). Mirrors vivado/build_vivado.tcl.
set ROOT [file dirname [file dirname [file normalize [info script]]]]
cd $ROOT

set FREQ  [lindex $argv 0]
set PART  [lindex $argv 1]
set tag falling_glucose_secded
set period [expr {1000.0 / double($FREQ)}]

file mkdir build/vivado reports/vivado

read_verilog -sv [list rtl/secded_decoder.v rtl/lyra_falling_glucose_core.v rtl/lyra_falling_glucose_top.v]

synth_design -top lyra_falling_glucose_top -part $PART
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
