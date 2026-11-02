// Synchronous wrapper around lyra_falling_glucose_core, mirroring
// lyra_zone_isolation_top.v's single-cycle pipeline shape for the second
// hardware-validated constraint.
`timescale 1ns/1ps
module lyra_falling_glucose_top (
    input             clk,
    input  signed [15:0] bg_rate_q,
    input  [21:0]     theta_codeword,
    output reg        block,
    output reg [1:0]  secded_status,
    output reg [16:0] margin_q_dbg,
    output reg [15:0] theta_corrected_dbg
);
    reg  signed [15:0] bg_r;
    reg  [21:0] cw_r;
    wire        block_c;
    wire [1:0]  status_c;
    wire [16:0] margin_c;
    wire [15:0] theta_c;

    always @(posedge clk) begin
        bg_r <= bg_rate_q;
        cw_r <= theta_codeword;
    end

    lyra_falling_glucose_core u_core (
        .bg_rate_q           (bg_r),
        .theta_codeword      (cw_r),
        .block               (block_c),
        .secded_status       (status_c),
        .margin_q_dbg        (margin_c),
        .theta_corrected_dbg (theta_c)
    );

    always @(posedge clk) begin
        block               <= block_c;
        secded_status       <= status_c;
        margin_q_dbg        <= margin_c;
        theta_corrected_dbg <= theta_c;
    end
endmodule
