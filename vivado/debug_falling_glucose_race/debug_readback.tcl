open_hw_manager
connect_hw_server
open_hw_target

set dev [lindex [get_hw_devices] 0]
set_property PROGRAM.FILE {D:/lyra_vio_build/build/vivado/lyra_vio_falling.bit} $dev
set_property PROBES.FILE  {D:/lyra_vio_build/build/vivado/lyra_vio_falling.ltx} $dev
program_hw_devices $dev
refresh_hw_device $dev

set vio [lindex [get_hw_vios -of_objects $dev] 0]
set p_fault [get_hw_probes fault_mask         -of_objects $vio]
set p_bg    [get_hw_probes bg_rate_q_raw      -of_objects $vio]
set p_block [get_hw_probes block              -of_objects $vio]
set p_stat  [get_hw_probes secded_status      -of_objects $vio]
set p_theta [get_hw_probes theta_corrected_dbg -of_objects $vio]
set p_statd [get_hw_probes secded_status_dbg   -of_objects $vio]

puts "All probes: [get_hw_probes -of_objects $vio]"

set_property OUTPUT_VALUE 000000 $p_fault
set_property OUTPUT_VALUE 0000   $p_bg
commit_hw_vio $vio
after 150
refresh_hw_vio $vio
puts "fault=0: theta_corrected_dbg=[get_property INPUT_VALUE $p_theta] status_dbg=[get_property INPUT_VALUE $p_statd] block=[get_property INPUT_VALUE $p_block] status=[get_property INPUT_VALUE $p_stat]"

close_hw_target
disconnect_hw_server
close_hw_manager
