open_hw_manager
connect_hw_server
open_hw_target

set dev [lindex [get_hw_devices] 0]
set_property PROGRAM.FILE {D:/lyra_vio_build/build/vivado/lyra_vio_falling.bit} $dev
set_property PROBES.FILE  {D:/lyra_vio_build/build/vivado/lyra_vio_falling.ltx} $dev
program_hw_devices $dev
refresh_hw_device $dev

set vio [lindex [get_hw_vios -of_objects $dev] 0]
set p_fault [get_hw_probes fault_mask      -of_objects $vio]
set p_bg    [get_hw_probes bg_rate_q_raw   -of_objects $vio]
set p_block [get_hw_probes block           -of_objects $vio]
set p_stat  [get_hw_probes secded_status   -of_objects $vio]

puts "p_bg properties:"
puts [report_property -return_string $p_bg]

proc drive_raw {fault bg_hex} {
    global vio p_fault p_bg
    set_property OUTPUT_VALUE [format %06x $fault] $p_fault
    set_property OUTPUT_VALUE $bg_hex $p_bg
    commit_hw_vio $vio
}
proc observe {} {
    global vio p_block p_stat p_bg
    refresh_hw_vio $vio
    return [list [get_property INPUT_VALUE $p_block] [get_property INPUT_VALUE $p_stat] [get_property OUTPUT_VALUE $p_bg]]
}

foreach {label hexval} {
    "bg=0x0000 (0)"      0000
    "bg=0x8000 (-4096)"  8000
    "bg=0xffff (-1/4096)" ffff
    "bg=0xff9c (-100/4096=-0.0244)" ff9c
    "bg=0xf666 (-2458/4096=-0.6001)" f666
} {
    drive_raw 0 $hexval
    after 150
    set r [observe]
    puts "$label -> readback_out=[lindex $r 2] block=[lindex $r 0] status=[lindex $r 1]"
}

close_hw_target
disconnect_hw_server
close_hw_manager
