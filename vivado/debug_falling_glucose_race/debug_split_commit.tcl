open_hw_manager
connect_hw_server
open_hw_target

set dev [lindex [get_hw_devices] 0]
set_property PROBES.FILE {D:/lyra_vio_build/build/vivado/lyra_vio_uart.ltx} $dev
refresh_hw_device $dev
set vio [lindex [get_hw_vios -of_objects $dev] 0]
set p_fault [get_hw_probes fault_mask    -of_objects $vio]
set p_bg    [get_hw_probes bg_rate_q_raw -of_objects $vio]
set p_block [get_hw_probes block         -of_objects $vio]
set p_stat  [get_hw_probes secded_status -of_objects $vio]

proc drive_split {fault bg} {
    global vio p_fault p_bg
    set bg_u [expr {$bg & 0xFFFF}]
    set_property OUTPUT_VALUE [format %06x $fault] $p_fault
    commit_hw_vio $vio
    set_property OUTPUT_VALUE [format %04x $bg_u]  $p_bg
    commit_hw_vio $vio
}
proc observe {} {
    global vio p_block p_stat
    refresh_hw_vio $vio
    return [list [get_property INPUT_VALUE $p_block] [get_property INPUT_VALUE $p_stat]]
}

foreach {label fault bg} {
    "baseline-SAFE"  0 0
    "baseline-BLOCK" 0 -2458
    "baseline-BLOCK-retry1" 0 -2458
    "baseline-BLOCK-retry2" 0 -2458
    "bit4-BLOCK"     16 -2458
    "bit4-BLOCK-retry" 16 -2458
} {
    drive_split $fault $bg
    after 300
    set r [observe]
    puts "$label fault=$fault bg=$bg -> block=[lindex $r 0] status=[lindex $r 1]"
}

close_hw_target
disconnect_hw_server
close_hw_manager
