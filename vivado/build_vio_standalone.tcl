# Standalone copy of Project-10(Lyra-FPGA)'s VIO build flow, run from a
# genuinely separate, paren-free directory (not a junction). The junction
# trick (D:\lyra_fpga) fixed the TOP-LEVEL vivado.bat/cmd.exe %cd% parsing bug
# fine, but debug-core (VIO/dbg_hub) generation spawns its OWN nested helper
# processes that apparently canonicalise the cwd through the reparse point
# back to the real "...(Lyra-FPGA)..." path internally, re-triggering the
# exact same bug one level deeper where the junction workaround can't reach.
# A real (non-reparse-point) directory has no canonical path to resolve back
# to, so this sidesteps the problem entirely. See Project-10 PRD Sec 9.x.

set PART [lindex $argv 0]
set FREQ [lindex $argv 1]
set period [expr {1000.0 / double($FREQ)}]

set ROOT "D:/lyra_vio_build"
cd $ROOT

create_project -force lyra_vio_proj build/vivado/lyra_vio_proj -part $PART

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
    rtl/secded_decoder.v rtl/lyra_zone_isolation_core.v rtl/lyra_zone_isolation_vio_top.v]
add_files -fileset constrs_1 -norecurse vivado/lyra_vio.xdc
set_property top lyra_zone_isolation_vio_top [current_fileset]

synth_design -top lyra_zone_isolation_vio_top -part $PART
create_clock -name sys_clk -period $period [get_ports clk] -add

opt_design
place_design
phys_opt_design
route_design

write_checkpoint -force build/vivado/lyra_vio_routed.dcp
report_utilization -file reports/vivado/lyra_vio_util.txt
report_timing_summary -delay_type min_max -file reports/vivado/lyra_vio_timing.txt

write_bitstream -force build/vivado/lyra_vio.bit
write_debug_probes -force build/vivado/lyra_vio.ltx

puts "DONE lyra_vio build part=$PART freq_mhz=$FREQ"
