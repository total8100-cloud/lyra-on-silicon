# Hybrid VIO(drive)+UART(observe) live fault-injection test for
# falling_glucose_guard. VIO still drives fault_mask/bg_rate_q (proven
# reliable in every prior run); block/secded_status are read back via VIO
# too (for a same-run cross-check) but the authoritative result comes from
# the UART log correlated by wall-clock timestamp, read independently by
# uart_reader.py running concurrently.
open_hw_manager
connect_hw_server
open_hw_target

set dev [lindex [get_hw_devices] 0]
set_property PROGRAM.FILE {D:/lyra_vio_build/build/vivado/lyra_vio_uart.bit} $dev
set_property PROBES.FILE  {D:/lyra_vio_build/build/vivado/lyra_vio_uart.ltx} $dev
program_hw_devices $dev
refresh_hw_device $dev

set vio [lindex [get_hw_vios -of_objects $dev] 0]
set p_fault [get_hw_probes fault_mask    -of_objects $vio]
set p_bg    [get_hw_probes bg_rate_q_raw -of_objects $vio]
set p_block [get_hw_probes block         -of_objects $vio]
set p_stat  [get_hw_probes secded_status -of_objects $vio]

proc drive {fault bg} {
    global vio p_fault p_bg
    set bg_u [expr {$bg & 0xFFFF}]
    set_property OUTPUT_VALUE [format %06x $fault] $p_fault
    set_property OUTPUT_VALUE [format %04x $bg_u]  $p_bg
    commit_hw_vio $vio
}
proc observe_vio {} {
    global vio p_block p_stat
    refresh_hw_vio $vio
    return [list [get_property INPUT_VALUE $p_block] [get_property INPUT_VALUE $p_stat]]
}

set BG_SAFE  0
set BG_BLOCK -2458

proc step {label fault bg} {
    global BG_SAFE
    drive $fault $bg
    after 400
    set r [observe_vio]
    set t [clock milliseconds]
    puts "STEP t=$t label=$label fault=$fault bg=$bg vio_block=[lindex $r 0] vio_status=[lindex $r 1]"
}

puts "TEST_START t=[clock milliseconds]"
step "baseline-SAFE"  0 $BG_SAFE
step "baseline-BLOCK" 0 $BG_BLOCK

foreach bit {0 4 8 9 11 12 16 20} {
    set mask [expr {1 << $bit}]
    step "bit${bit}-SAFE"  $mask $BG_SAFE
    step "bit${bit}-BLOCK" $mask $BG_BLOCK
}

foreach pair {{4 5} {11 12} {0 21}} {
    set b0 [lindex $pair 0]; set b1 [lindex $pair 1]
    set mask [expr {(1 << $b0) | (1 << $b1)}]
    step "pair${b0}_${b1}-SAFE" $mask $BG_SAFE
}
puts "TEST_END t=[clock milliseconds]"

close_hw_target
disconnect_hw_server
close_hw_manager
