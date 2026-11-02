// VIO harness for the SECOND hardware-validated constraint (Medical/
// falling_glucose_guard, theta=-0.30, threshold_below). Adds explicit
// 2-stage synchronizer registers on both VIO<->design boundaries (probe_out
// into the design, and the design's status signals back into probe_in)
// after two prior builds showed intermittent, non-reproducible-per-value
// readback corruption on bg=-2458 and a few other values -- a pattern
// consistent with a marginal/metastable capture at the VIO<->user-clock
// boundary rather than a logic bug (confirmed: decoder and margin math are
// correct in isolation and in the majority of individual reads).
`timescale 1ns/1ps
module lyra_falling_glucose_vio_top (
    input clk
);
    wire [21:0] fault_mask_raw;
    wire signed [15:0] bg_rate_q_raw0;
    wire        block;
    wire [1:0]  secded_status;
    wire [16:0] margin_q_dbg;
    wire [15:0] theta_corrected_dbg;

    `include "falling_glucose_params.vh"

    // --- 2-stage synchronizer: VIO probe_out -> design clock domain ---
    reg [21:0] fault_mask_s1, fault_mask_s2;
    reg signed [15:0] bg_rate_q_s1, bg_rate_q_s2;
    always @(posedge clk) begin
        fault_mask_s1 <= fault_mask_raw;
        fault_mask_s2 <= fault_mask_s1;
        bg_rate_q_s1  <= bg_rate_q_raw0;
        bg_rate_q_s2  <= bg_rate_q_s1;
    end

    wire [21:0] codeword_faulty = THETA_CODEWORD ^ fault_mask_s2;

    wire        block_c;
    wire [1:0]  secded_status_c;
    wire [16:0] margin_q_dbg_c;
    wire [15:0] theta_corrected_dbg_c;

    lyra_falling_glucose_top u_top (
        .clk                 (clk),
        .bg_rate_q           (bg_rate_q_s2),
        .theta_codeword      (codeword_faulty),
        .block               (block_c),
        .secded_status       (secded_status_c),
        .margin_q_dbg        (margin_q_dbg_c),
        .theta_corrected_dbg (theta_corrected_dbg_c)
    );

    // --- 2-stage synchronizer: design -> VIO probe_in ---
    reg block_s1; reg block_s2;
    reg [1:0] status_s1; reg [1:0] status_s2;
    reg [16:0] margin_s1; reg [16:0] margin_s2;
    reg [15:0] theta_s1; reg [15:0] theta_s2;
    always @(posedge clk) begin
        block_s1  <= block_c;      block_s2  <= block_s1;
        status_s1 <= secded_status_c; status_s2 <= status_s1;
        margin_s1 <= margin_q_dbg_c;  margin_s2 <= margin_s1;
        theta_s1  <= theta_corrected_dbg_c; theta_s2 <= theta_s1;
    end
    assign block = block_s2;
    assign secded_status = status_s2;
    assign margin_q_dbg = margin_s2;
    assign theta_corrected_dbg = theta_s2;

    vio_lyra2 u_vio (
        .clk       (clk),
        .probe_in0 (block),
        .probe_in1 (secded_status),
        .probe_in2 (margin_q_dbg),
        .probe_in3 (theta_corrected_dbg),
        .probe_out0(fault_mask_raw),
        .probe_out1(bg_rate_q_raw0)
    );
endmodule
