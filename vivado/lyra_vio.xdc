## Arty S7 constraints for lyra_zone_isolation_vio_top -- only the clock is a
## real pin; every other signal is reached via the VIO debug core over JTOT,
## so no other port exists at this level. Same Rev. E pinout / config block as
## Project-08's vivado/aria_arty_s7.xdc (see its header note on board revision).
set_property -dict { PACKAGE_PIN F14  IOSTANDARD LVCMOS33 } [get_ports { clk }]; #Sch=uclk

set_property BITSTREAM.CONFIG.CONFIGRATE 50 [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property CFGBVS VCCO [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property CONFIG_MODE SPIx4 [current_design]
set_property INTERNAL_VREF 0.675 [get_iobanks 34]
