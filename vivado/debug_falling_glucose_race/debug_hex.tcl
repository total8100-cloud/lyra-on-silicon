set bg -2458
set bg_u [expr {$bg & 0xFFFF}]
puts "bg=$bg bg_u=$bg_u hex=[format %04x $bg_u]"
