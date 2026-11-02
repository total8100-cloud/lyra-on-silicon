open_hw_manager
connect_hw_server
open_hw_target

set dev [lindex [get_hw_devices] 0]
set_property PROGRAM.FILE {D:/lyra_vio_build/build/vivado/lyra_vio_falling.bit} $dev
set_property PROBES.FILE  {D:/lyra_vio_build/build/vivado/lyra_vio_falling.ltx} $dev
program_hw_devices $dev
refresh_hw_device $dev

set vio [lindex [get_hw_vios -of_objects $dev] 0]
set p_fault  [get_hw_probes fault_mask         -of_objects $vio]
set p_bg     [get_hw_probes bg_rate_q_raw      -of_objects $vio]
set p_block  [get_hw_probes block              -of_objects $vio]
set p_stat   [get_hw_probes secded_status      -of_objects $vio]
set p_margin [get_hw_probes margin_q_dbg       -of_objects $vio]
set p_theta  [get_hw_probes theta_corrected_dbg -of_objects $vio]

proc drive {fault bg} {
    global vio p_fault p_bg
    set bg_u [expr {$bg & 0xFFFF}]
    set_property OUTPUT_VALUE [format %06x $fault] $p_fault
    set_property OUTPUT_VALUE [format %04x $bg_u]  $p_bg
    commit_hw_vio $vio
}
proc observe {} {
    global vio p_block p_stat p_margin p_theta
    refresh_hw_vio $vio
    return [list [get_property INPUT_VALUE $p_block] [get_property INPUT_VALUE $p_stat] \
                 [get_property INPUT_VALUE $p_margin] [get_property INPUT_VALUE $p_theta]]
}

foreach bg {-500 -1000 -1230 -1300 -1500 -2000 -2458 -2458 -3000 -3500 -4000} {
    drive 0 $bg
    after 150
    set r [observe]
    if {[lindex $r 3] eq "fb33"} { set theta_ok "OK" } else { set theta_ok "BAD" }
    puts "bg=$bg -> block=[lindex $r 0] status=[lindex $r 1] margin=[lindex $r 2] theta=[lindex $r 3] ($theta_ok)"
}

close_hw_target
disconnect_hw_server
close_hw_manager
