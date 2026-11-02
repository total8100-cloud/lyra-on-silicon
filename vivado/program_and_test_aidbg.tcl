# Hardware-in-the-loop AI-detector validation: collect REAL, silicon-decoded
# theta values for a sample of triple-bit fault injections against
# zone_isolation (theta=0.30, Q4.12/16-bit), for feeding into the
# software-trained IsolationForest detector (pilot/lyra_ai_drift_detector.py).
# zone_isolation's VIO readback has been reliable throughout this project
# (unlike falling_glucose's wider signed probe), so no UART cross-check needed.
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

proc drive {fault zd auth} {
    global vio p_fault p_zd p_auth
    set_property OUTPUT_VALUE [format %06x $fault] $p_fault
    set_property OUTPUT_VALUE [format %04x $zd]    $p_zd
    set_property OUTPUT_VALUE [format %01x $auth]  $p_auth
    commit_hw_vio $vio
}
proc observe {} {
    global vio p_block p_stat p_theta
    refresh_hw_vio $vio
    return [list [get_property INPUT_VALUE $p_block] [get_property INPUT_VALUE $p_stat] [get_property INPUT_VALUE $p_theta]]
}

set ZD_SAFE 410

# baseline (no fault)
drive 0 $ZD_SAFE 0
after 200
set r [observe]
puts "SAMPLE label=baseline fault=0 theta_hex=[lindex $r 2] status=[lindex $r 1] block=[lindex $r 0]"

# deterministic pseudo-random sample of 60 distinct triples from the 21-bit codeword
set seed 12345
proc nextrand {} {
    global seed
    set seed [expr {(1103515245 * $seed + 12345) % 2147483648}]
    return $seed
}
set seen {}
set n 0
while {$n < 60} {
    set a [expr {[nextrand] % 21}]
    set b [expr {[nextrand] % 21}]
    set c [expr {[nextrand] % 21}]
    if {$a == $b || $b == $c || $a == $c} { continue }
    set key [lsort -integer [list $a $b $c]]
    if {[lsearch -exact $seen $key] >= 0} { continue }
    lappend seen $key
    set mask [expr {(1 << $a) | (1 << $b) | (1 << $c)}]
    drive $mask $ZD_SAFE 0
    after 200
    set r [observe]
    puts "SAMPLE label=triple_${a}_${b}_${c} fault=$mask theta_hex=[lindex $r 2] status=[lindex $r 1] block=[lindex $r 0]"
    incr n
}

puts "AIDBG_COLLECTION_DONE n=$n"

close_hw_target
disconnect_hw_server
close_hw_manager
