# Same standalone paren-free-directory VIO flow as build_vio.tcl, but wraps
# the REGISTERED lyra_zone_isolation_top (the exact netlist whose LUT 66 /
# FF 42 / Fmax~200MHz numbers were reported in Project-10 PRD Sec 9.6) instead
# of the bare combinational core -- closes the gap flagged 2026-10-15 (that
# build was placed/routed for the real part but never flashed and run).

set PART [lindex $argv 0]
set FREQ [lindex $argv 1]
set period [expr {1000.0 / double($FREQ)}]

set ROOT "D:/lyra_vio_build"
cd $ROOT

create_project -force lyra_vio_reg_proj build/vivado/lyra_vio_reg_proj -part $PART

create_ip -name vio -vendor xilinx.com -library ip -version 3.0 -module_name vio_lyra
set_property -dict [list \
    CONFIG.C_NUM_PROBE_OUT {3} \
    CONFIG.C_PROBE_OUT0_WIDTH {22} \
    CONFIG.C_PROBE_OUT1_WIDTH {16} \
    CONFIG.C_PROBE_OUT2_WIDTH {1} \
    CONFIG.C_NUM_PROBE_IN {2} \
    CONFIG.C_PROBE_IN0_WIDTH {1} \
    CONFIG.C_PROBE_IN1_WIDTH {2} \
] [get_ips vio_lyra]
generate_target all [get_files -of_objects [get_ips vio_lyra]]
synth_ip [get_ips vio_lyra]

add_files -norecurse [list \
    rtl/secded_decoder.v rtl/lyra_zone_isolation_core.v \
    rtl/lyra_zone_isolation_top.v rtl/lyra_zone_isolation_top_vio.v]
add_files -fileset constrs_1 -norecurse vivado/lyra_vio.xdc
set_property top lyra_zone_isolation_top_vio [current_fileset]

synth_design -top lyra_zone_isolation_top_vio -part $PART
create_clock -name sys_clk -period $period [get_ports clk] -add

opt_design
place_design
phys_opt_design
route_design

write_checkpoint -force build/vivado/lyra_vio_reg_routed.dcp
report_utilization -file reports/vivado/lyra_vio_reg_util.txt
report_timing_summary -delay_type min_max -file reports/vivado/lyra_vio_reg_timing.txt

write_bitstream -force build/vivado/lyra_vio_reg.bit
write_debug_probes -force build/vivado/lyra_vio_reg.ltx

puts "DONE lyra_vio_reg build part=$PART freq_mhz=$FREQ"
