# Hardware-in-the-loop validation of the behavioral integrity monitor (PRD Sec 9.19):
# replays a stimulus stream (pilot/lyra_behavioral_hil.py gen) through the real
# SECDED-protected zone_isolation shield on the Arty S7-50 and logs, per decision,
# the silicon's BLOCK bit (the only thing the monitor uses), plus secded_status and
# the decoded theta (logged for the B2 baseline / cross-checking only).
# Line format in: "<segment> <fault_mask> <zone_distance_q> <auth_state>"
# Line format out: "D <segment> <index> <block> <status> <theta_hex>"
set STIM [lindex $argv 0]

open_hw_manager
connect_hw_server
open_hw_target

set dev [lindex [get_hw_devices] 0]
set_property PROGRAM.FILE {D:/lyra_vio_build/build/vivado/lyra_vio_aidbg.bit} $dev
set_property PROBES.FILE  {D:/lyra_vio_build/build/vivado/lyra_vio_aidbg.ltx} $dev
program_hw_devices $dev
refresh_hw_device $dev

set vio [lindex [get_hw_vios -of_objects $dev] 0]
set p_fault [get_hw_probes fault_mask           -of_objects $vio]
set p_zd    [get_hw_probes zone_distance_q_raw  -of_objects $vio]
set p_auth  [get_hw_probes auth_state           -of_objects $vio]
set p_block [get_hw_probes block                -of_objects $vio]
set p_stat  [get_hw_probes secded_status        -of_objects $vio]
set p_theta [get_hw_probes theta_corrected_dbg  -of_objects $vio]

set fh [open $STIM r]
set prev ""
set idx 0
set t0 [clock milliseconds]
while {[gets $fh line] >= 0} {
    if {[string trim $line] eq ""} { continue }
    lassign $line seg mask zd auth
    if {$seg ne $prev} { set idx 0; set prev $seg }
    set_property OUTPUT_VALUE [format %06x $mask] $p_fault
    set_property OUTPUT_VALUE [format %04x $zd]   $p_zd
    set_property OUTPUT_VALUE [format %01x $auth] $p_auth
    commit_hw_vio $vio
    refresh_hw_vio $vio
    puts "D $seg $idx [get_property INPUT_VALUE $p_block] [get_property INPUT_VALUE $p_stat] [get_property INPUT_VALUE $p_theta]"
    incr idx
}
close $fh
puts "BEHAVIORAL_HIL_DONE elapsed_ms=[expr {[clock milliseconds] - $t0}]"

close_hw_target
disconnect_hw_server
close_hw_manager
