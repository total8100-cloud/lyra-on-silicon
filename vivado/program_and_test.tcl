# Real-hardware fault injection: program the Arty S7 with the VIO-instrumented
# bitstream and drive/observe the SECDED-protected zone_isolation core live,
# over JTAG, scripted (non-interactive) -- Project-10 PRD Sec 9.8 item 1.
open_hw_manager
connect_hw_server
open_hw_target

set dev [lindex [get_hw_devices] 0]
puts "Using hw_device: $dev"
set_property PROGRAM.FILE {D:/lyra_vio_build/build/vivado/lyra_vio.bit} $dev
set_property PROBES.FILE  {D:/lyra_vio_build/build/vivado/lyra_vio.ltx} $dev
program_hw_devices $dev
refresh_hw_device $dev

set vio [lindex [get_hw_vios -of_objects $dev] 0]
puts "Using hw_vio: $vio"

set all_probes [get_hw_probes -of_objects $vio]
puts "Available probes: $all_probes"

# Vivado named each hw_probe after the RTL net it connects to, not the generic
# probe_outN/probe_inN names used at instantiation.
set p_fault [get_hw_probes fault_mask       -of_objects $vio]
set p_zd    [get_hw_probes zone_distance_q  -of_objects $vio]
set p_auth  [get_hw_probes auth_state       -of_objects $vio]
set p_block [get_hw_probes block            -of_objects $vio]
set p_stat  [get_hw_probes secded_status    -of_objects $vio]
puts "p_fault=$p_fault p_zd=$p_zd p_auth=$p_auth p_block=$p_block p_stat=$p_stat"

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
    set b [get_property INPUT_VALUE $p_block]
    set s [get_property INPUT_VALUE $p_stat]
    return [list $b $s]
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

drive 0 $ZD_SAFE 0;  after 50; check "baseline SAFE " 0 [observe]
drive 0 $ZD_BLOCK 0; after 50; check "baseline BLOCK" 1 [observe]

foreach bit {0 4 8 11 12 16 20 21} {
    set mask [expr {1 << $bit}]
    drive $mask $ZD_SAFE 0;  after 50; check "bit$bit SAFE   " 0 [observe]
    drive $mask $ZD_BLOCK 0; after 50; check "bit$bit BLOCK  " 1 [observe]
}

foreach pair {{4 5} {11 12} {0 21}} {
    set b0 [lindex $pair 0]
    set b1 [lindex $pair 1]
    set mask [expr {(1 << $b0) | (1 << $b1)}]
    drive $mask $ZD_SAFE 0; after 50
    set r [observe]
    puts "double($b0,$b1) SAFE: block=[lindex $r 0] status=[lindex $r 1] (expect block=1, status=2 -- forced fail-safe)"
    if {[lindex $r 0] != 1} { incr errors }
}

if {$errors == 0} {
    puts "REAL_HW_FAULT_INJECTION PASS"
} else {
    puts "REAL_HW_FAULT_INJECTION FAIL ($errors mismatches)"
}

close_hw_target
disconnect_hw_server
close_hw_manager
