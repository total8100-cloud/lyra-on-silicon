## Arty S7 constraints for lyra_falling_glucose_uart_top -- clock + the
## onboard USB-UART bridge TX pin; every other signal is still reached via
## the VIO debug core over JTAG, same as lyra_vio.xdc.
set_property -dict { PACKAGE_PIN F14  IOSTANDARD LVCMOS33 } [get_ports { clk }]; #Sch=uclk
set_property -dict { PACKAGE_PIN R12  IOSTANDARD LVCMOS33 } [get_ports { uart_rxd_out }]; #Sch=uart_rxd_out (FPGA TX -> host RX)

set_property BITSTREAM.CONFIG.CONFIGRATE 50 [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property CFGBVS VCCO [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property CONFIG_MODE SPIx4 [current_design]
set_property INTERNAL_VREF 0.675 [get_iobanks 34]
