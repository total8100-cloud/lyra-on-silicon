# Real-hardware fault injection for the SECOND validated constraint
# (Medical/falling_glucose_guard, theta=-0.30, threshold_below, negative
# theta, single-input expression -- deliberately different from
# zone_isolation on every axis). Mirrors program_and_test_registered.tcl.
open_hw_manager
connect_hw_server
open_hw_target

set dev [lindex [get_hw_devices] 0]
puts "Using hw_device: $dev"
set_property PROGRAM.FILE {D:/lyra_vio_build/build/vivado/lyra_vio_falling.bit} $dev
set_property PROBES.FILE  {D:/lyra_vio_build/build/vivado/lyra_vio_falling.ltx} $dev
program_hw_devices $dev
refresh_hw_device $dev

set vio [lindex [get_hw_vios -of_objects $dev] 0]
puts "Using hw_vio: $vio"
set all_probes [get_hw_probes -of_objects $vio]
puts "Available probes: $all_probes"

set p_fault [get_hw_probes fault_mask      -of_objects $vio]
set p_bg    [get_hw_probes bg_rate_q_raw   -of_objects $vio]
set p_block [get_hw_probes block           -of_objects $vio]
set p_stat  [get_hw_probes secded_status   -of_objects $vio]

proc drive {fault bg} {
    global vio p_fault p_bg
    set bg_u [expr {$bg & 0xFFFF}]
    set_property OUTPUT_VALUE [format %06x $fault] $p_fault
    set_property OUTPUT_VALUE [format %04x $bg_u]  $p_bg
    commit_hw_vio $vio
}
proc observe {} {
    global vio p_block p_stat
    refresh_hw_vio $vio
    return [list [get_property INPUT_VALUE $p_block] [get_property INPUT_VALUE $p_stat]]
}

# Q4.12: BG_SAFE=0 (expr=0.0 >= theta=-0.30 -> safe), BG_BLOCK=-2458 (expr=-0.60 < -0.30 -> block)
set BG_SAFE  0
set BG_BLOCK -2458
set errors 0

proc check {label expect_block got} {
    global errors
    set b [lindex $got 0]
    set s [lindex $got 1]
    set ok [expr {$b == $expect_block}]
    if {!$ok} { incr errors }
    puts "$label: block=$b status=$s (expect block=$expect_block) [expr {$ok ? "OK" : "MISMATCH"}]"
}

drive 0 $BG_SAFE;  after 100; check "baseline SAFE " 0 [observe]
drive 0 $BG_BLOCK; after 100; check "baseline BLOCK" 1 [observe]

foreach bit {0 4 8 9 11 12 16 20 21} {
    set mask [expr {1 << $bit}]
    drive $mask $BG_SAFE;  after 100; check "bit$bit SAFE   " 0 [observe]
    drive $mask $BG_BLOCK; after 100; check "bit$bit BLOCK  " 1 [observe]
}

foreach pair {{4 5} {11 12} {0 21}} {
    set b0 [lindex $pair 0]; set b1 [lindex $pair 1]
    set mask [expr {(1 << $b0) | (1 << $b1)}]
    drive $mask $BG_SAFE; after 100
    set r [observe]
    puts "double($b0,$b1) SAFE: block=[lindex $r 0] status=[lindex $r 1] (expect block=1, status=2)"
    if {[lindex $r 0] != 1} { incr errors }
}

if {$errors == 0} {
    puts "REAL_HW_FAULT_INJECTION_FALLING_GLUCOSE PASS"
} else {
    puts "REAL_HW_FAULT_INJECTION_FALLING_GLUCOSE FAIL ($errors mismatches)"
}

close_hw_target
disconnect_hw_server
close_hw_manager
