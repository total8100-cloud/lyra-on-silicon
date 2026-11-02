# Real-hardware fault injection against the REGISTERED lyra_zone_isolation_top
# (the exact LUT-66/FF-42 netlist reported in PRD Sec 9.6) -- closes the gap
# flagged 2026-10-15 between "placed/routed for the real part" and "actually
# run on the chip."
open_hw_manager
connect_hw_server
open_hw_target

set dev [lindex [get_hw_devices] 0]
puts "Using hw_device: $dev"
set_property PROGRAM.FILE {D:/lyra_vio_build/build/vivado/lyra_vio_reg.bit} $dev
set_property PROBES.FILE  {D:/lyra_vio_build/build/vivado/lyra_vio_reg.ltx} $dev
program_hw_devices $dev
refresh_hw_device $dev

set vio [lindex [get_hw_vios -of_objects $dev] 0]
puts "Using hw_vio: $vio"
set all_probes [get_hw_probes -of_objects $vio]
puts "Available probes: $all_probes"

set p_fault [get_hw_probes fault_mask           -of_objects $vio]
set p_zd    [get_hw_probes zone_distance_q_raw  -of_objects $vio]
set p_auth  [get_hw_probes auth_state           -of_objects $vio]
set p_block [get_hw_probes block                -of_objects $vio]
set p_stat  [get_hw_probes secded_status        -of_objects $vio]

proc drive {fault zd auth} {
    global vio p_fault p_zd p_auth
    set_property OUTPUT_VALUE [format %06x $fault] $p_fault
    set_property OUTPUT_VALUE [format %04x $zd]    $p_zd
    set_property OUTPUT_VALUE [format %01x $auth]  $p_auth
    commit_hw_vio $vio
}
proc observe {} {
    global vio p_block p_stat
    refresh_hw_vio $vio
    return [list [get_property INPUT_VALUE $p_block] [get_property INPUT_VALUE $p_stat]]
}

set ZD_SAFE  410
set ZD_BLOCK 2048
set errors 0

proc check {label expect_block got} {
    global errors
    set b [lindex $got 0]
    set s [lindex $got 1]
    set ok [expr {$b == $expect_block}]
    if {!$ok} { incr errors }
    puts "$label: block=$b status=$s (expect block=$expect_block) [expr {$ok ? "OK" : "MISMATCH"}]"
}

# this design is REGISTERED (one clock-cycle pipeline) -- VIO's clk net is
# not running at a fixed observable rate from Tcl, so give it a bit more
# settle time than the combinational-core test did.
drive 0 $ZD_SAFE 0;  after 100; check "baseline SAFE " 0 [observe]
drive 0 $ZD_BLOCK 0; after 100; check "baseline BLOCK" 1 [observe]

foreach bit {0 4 8 11 12 16 20 21} {
    set mask [expr {1 << $bit}]
    drive $mask $ZD_SAFE 0;  after 100; check "bit$bit SAFE   " 0 [observe]
    drive $mask $ZD_BLOCK 0; after 100; check "bit$bit BLOCK  " 1 [observe]
}

foreach pair {{4 5} {11 12} {0 21}} {
    set b0 [lindex $pair 0]; set b1 [lindex $pair 1]
    set mask [expr {(1 << $b0) | (1 << $b1)}]
    drive $mask $ZD_SAFE 0; after 100
    set r [observe]
    puts "double($b0,$b1) SAFE: block=[lindex $r 0] status=[lindex $r 1] (expect block=1, status=2)"
    if {[lindex $r 0] != 1} { incr errors }
}

if {$errors == 0} {
    puts "REAL_HW_FAULT_INJECTION_REGISTERED PASS"
} else {
    puts "REAL_HW_FAULT_INJECTION_REGISTERED FAIL ($errors mismatches)"
}

close_hw_target
disconnect_hw_server
close_hw_manager
