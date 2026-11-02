# Standalone VIO build for the AI-detector hardware-in-the-loop extension:
# same registered lyra_zone_isolation_top as build_vio_registered.tcl, plus a
# theta_corrected_dbg readback probe so a host script can collect real
# silicon-measured decoded-theta values under triple-bit fault injection.

set PART [lindex $argv 0]
set FREQ [lindex $argv 1]
set period [expr {1000.0 / double($FREQ)}]

set ROOT "D:/lyra_vio_build"
cd $ROOT

file delete -force build/vivado/lyra_vio_aidbg_proj
create_project -force lyra_vio_aidbg_proj build/vivado/lyra_vio_aidbg_proj -part $PART

create_ip -name vio -vendor xilinx.com -library ip -version 3.0 -module_name vio_lyra3
set_property -dict [list \
    CONFIG.C_NUM_PROBE_OUT {3} \
    CONFIG.C_PROBE_OUT0_WIDTH {22} \
    CONFIG.C_PROBE_OUT1_WIDTH {16} \
    CONFIG.C_PROBE_OUT2_WIDTH {1} \
    CONFIG.C_NUM_PROBE_IN {3} \
    CONFIG.C_PROBE_IN0_WIDTH {1} \
    CONFIG.C_PROBE_IN1_WIDTH {2} \
    CONFIG.C_PROBE_IN2_WIDTH {16} \
] [get_ips vio_lyra3]
generate_target all [get_files -of_objects [get_ips vio_lyra3]]
synth_ip [get_ips vio_lyra3]

add_files -norecurse [list \
    rtl/secded_decoder.v rtl/lyra_zone_isolation_core.v \
    rtl/lyra_zone_isolation_top.v rtl/lyra_zone_isolation_top_vio_aidbg.v]
add_files -fileset constrs_1 -norecurse vivado/lyra_vio.xdc
set_property top lyra_zone_isolation_top_vio_aidbg [current_fileset]

synth_design -top lyra_zone_isolation_top_vio_aidbg -part $PART
create_clock -name sys_clk -period $period [get_ports clk] -add

opt_design
place_design
phys_opt_design
route_design

write_checkpoint -force build/vivado/lyra_vio_aidbg_routed.dcp
report_utilization -file reports/vivado/lyra_vio_aidbg_util.txt

write_bitstream -force build/vivado/lyra_vio_aidbg.bit
write_debug_probes -force build/vivado/lyra_vio_aidbg.ltx

puts "DONE lyra_vio_aidbg build part=$PART freq_mhz=$FREQ"
