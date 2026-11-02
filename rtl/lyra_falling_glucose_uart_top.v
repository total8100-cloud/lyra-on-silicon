// Hybrid harness: VIO still drives fault_mask/bg_rate_q (the control path
// already proven reliable -- committed OUTPUT_VALUE always read back
// correctly in every prior diagnostic run), but block/secded_status are now
// also continuously streamed out over the Arty S7's onboard USB-UART bridge
// (115200 8N1), bypassing the JTAG/VIO debug-probe readback path entirely
// for the result side. VIO's own probe_in readback of block/status is kept
// too, purely for a same-run cross-check against the UART stream.
`timescale 1ns/1ps
module lyra_falling_glucose_uart_top (
    input  clk,
    output uart_rxd_out  // Arty S7 PACKAGE_PIN R12 (FPGA TX -> host RX)
);
    wire [21:0] fault_mask;
    wire signed [15:0] bg_rate_q_raw;
    wire        block;
    wire [1:0]  secded_status;
    wire [16:0] margin_q_dbg;
    wire [15:0] theta_corrected_dbg;

    `include "falling_glucose_params.vh"

    wire [21:0] codeword_faulty = THETA_CODEWORD ^ fault_mask;

    lyra_falling_glucose_top u_top (
        .clk                 (clk),
        .bg_rate_q           (bg_rate_q_raw),
        .theta_codeword      (codeword_faulty),
        .block               (block),
        .secded_status       (secded_status),
        .margin_q_dbg        (margin_q_dbg),
        .theta_corrected_dbg (theta_corrected_dbg)
    );

    vio_lyra2 u_vio (
        .clk       (clk),
        .probe_in0 (block),
        .probe_in1 (secded_status),
        .probe_out0(fault_mask),
        .probe_out1(bg_rate_q_raw)
    );

    uart_tx #(.CLK_FREQ_HZ(12_000_000), .BAUD(115_200)) u_uart (
        .clk  (clk),
        .data ({5'b0, secded_status, block}),
        .txd  (uart_rxd_out)
    );
endmodule
