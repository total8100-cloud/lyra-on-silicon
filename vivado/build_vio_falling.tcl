# Standalone VIO build for the SECOND hardware-validated constraint
# (Medical/falling_glucose_guard). Same paren-free-directory requirement as
# build_vio_registered.tcl (debug-core generation needs a non-reparse-point
# directory, not the D:\lyra_fpga junction).

set PART [lindex $argv 0]
set FREQ [lindex $argv 1]
set period [expr {1000.0 / double($FREQ)}]

set ROOT "D:/lyra_vio_build"
cd $ROOT

file delete -force build/vivado/lyra_vio_falling_proj
create_project -force lyra_vio_falling_proj build/vivado/lyra_vio_falling_proj -part $PART

create_ip -name vio -vendor xilinx.com -library ip -version 3.0 -module_name vio_lyra2
set_property -dict [list \
    CONFIG.C_NUM_PROBE_OUT {2} \
    CONFIG.C_PROBE_OUT0_WIDTH {22} \
    CONFIG.C_PROBE_OUT1_WIDTH {16} \
    CONFIG.C_NUM_PROBE_IN {4} \
    CONFIG.C_PROBE_IN0_WIDTH {1} \
    CONFIG.C_PROBE_IN1_WIDTH {2} \
    CONFIG.C_PROBE_IN2_WIDTH {17} \
    CONFIG.C_PROBE_IN3_WIDTH {16} \
] [get_ips vio_lyra2]
generate_target all [get_files -of_objects [get_ips vio_lyra2]]
synth_ip [get_ips vio_lyra2]

add_files -norecurse [list \
    rtl/secded_decoder.v rtl/lyra_falling_glucose_core.v \
    rtl/lyra_falling_glucose_top.v rtl/lyra_falling_glucose_vio_top.v]
add_files -fileset constrs_1 -norecurse vivado/lyra_vio.xdc
set_property top lyra_falling_glucose_vio_top [current_fileset]

synth_design -top lyra_falling_glucose_vio_top -part $PART
create_clock -name sys_clk -period $period [get_ports clk] -add

opt_design
place_design
phys_opt_design
route_design

write_checkpoint -force build/vivado/lyra_vio_falling_routed.dcp
report_utilization -file reports/vivado/lyra_vio_falling_util.txt

write_bitstream -force build/vivado/lyra_vio_falling.bit
write_debug_probes -force build/vivado/lyra_vio_falling.ltx

puts "DONE lyra_vio_falling build part=$PART freq_mhz=$FREQ"
