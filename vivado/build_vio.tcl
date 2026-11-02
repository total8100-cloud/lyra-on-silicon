# Non-project batch flow: synthesise lyra_zone_isolation_vio_top (SECDED core
# + VIO debug core) and produce a bitstream + debug-probes (.ltx) file for
# real-hardware fault injection on the Arty S7 (PRD 9.8 item 1).
#
# Same parenthesized-path workaround as Project-08 / this project's other
# Vivado scripts -- launched via scripts/vio_flow.ps1 from the paren-free
# Vivado bin dir, through the D:\lyra_fpga junction.
#
# Usage: scripts/vio_flow.ps1 -Freq 50 -Part xc7s50csga324-1

set ROOT [file dirname [file dirname [file normalize [info script]]]]
cd $ROOT

set FREQ [lindex $argv 0]
set PART [lindex $argv 1]
set period [expr {1000.0 / double($FREQ)}]

file mkdir build/vivado reports/vivado vivado/ip

# A real (non-in-memory) project is required here: -in_memory projects were
# observed to register create_ip's IP object but never actually write its
# generated sources to disk (generate_target had nothing to act on, synth_ip's
# internal out-of-context synth_design then failed with "No HDL sources found
# in project") -- a real project directory gives IP core-container generation
# somewhere concrete to land. create_project additionally validates the FULL
# PROJECT PATH for illegal characters (not just cwd, unlike plain `cd`), so it
# must be created through the paren-free D:\lyra_fpga junction, not $ROOT.
file delete -force /lyra_fpga/build/vivado/lyra_vio_proj
create_project -force lyra_vio_proj /lyra_fpga/build/vivado/lyra_vio_proj -part $PART

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
# Use -of_objects rather than a hand-written path string: create_ip's actual
# managed-IP output location for an in-memory project is not guaranteed to be
# the literal "vivado/ip/vio_lyra/..." this script might otherwise assume (a
# hardcoded path here produced "No files matched" against the real location).
generate_target all [get_files -of_objects [get_ips vio_lyra]]

# synth_design on the top module does NOT automatically synthesize a vendor
# IP's own (encrypted) sources -- that only happens via the project's synth_1
# RUN machinery (launch_runs) in the GUI flow. The documented batch-mode
# equivalent is to synthesize the IP out-of-context explicitly first; its
# resulting checkpoint is then picked up automatically when the top-level
# synth_design elaborates the instantiating module.
synth_ip [get_ips vio_lyra]

# create_ip already auto-adds the .xci to this in-memory project's sources_1
# fileset -- use add_files (project-mode source management) for the plain RTL
# too, rather than mixing in read_verilog/read_ip (non-project-mode commands),
# which left the IP's own encrypted reference-model files unresolved ("module
# 'vio_lyra' not found", then "cannot open include file ... Error generated
# from encrypted envelope" when read_verilog was pointed at them manually).
# Letting synth_design run in project mode resolves the IP's own include
# paths correctly.
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
