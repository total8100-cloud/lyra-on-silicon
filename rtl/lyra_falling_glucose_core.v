// H_i evaluation core for Lyra's Medical-domain `falling_glucose_guard`
// constraint (ADA trend-arrow protocol): "block standard bolus when BG is
// rapidly falling." Compiled form (auto_compiler.py, threshold_below):
//   margin = expr - theta ;  BLOCK iff margin < 0   (expr = bg_rate directly)
// Chosen as the SECOND hardware-validated constraint specifically because
// it differs from zone_isolation (Sec. 10/11 of the paper) on every axis
// that matters: theta is NEGATIVE (-0.30, vs +0.30), the comparison
// direction is reversed (threshold_below vs threshold_above), and the
// compiled expression is a single state variable with no mux at all (no
// auth_state-style binary gating term) -- the simplest possible compiled
// circuit shape, as a lower bound on how little logic separates "safe"
// from "always-SAFE after a fault."
`timescale 1ns/1ps
module lyra_falling_glucose_core (
    input  signed [15:0] bg_rate_q,     // Q4.12 signed fixed point, valid range [-1,1)
    input  [21:0] theta_codeword,       // SECDED-protected compiled threshold (theta=-0.30)
    output        block,                // 1 = BLOCK (violation or integrity failure)
    output [1:0]  secded_status,        // 00 ok, 01 corrected, 10 uncorrectable
    output [16:0] margin_q_dbg,         // diagnostic-only tap, inert for block/status
    output [15:0] theta_corrected_dbg   // diagnostic-only tap, inert for block/status
);
    wire [15:0] theta_corrected;

    secded_decoder #(.K(16), .M(5)) u_decoder (
        .codeword (theta_codeword),
        .data_out (theta_corrected),
        .status   (secded_status)
    );

    wire signed [16:0] bg_s    = {bg_rate_q[15], bg_rate_q};
    wire signed [16:0] theta_s = {theta_corrected[15], theta_corrected};
    wire signed [16:0] margin_q = bg_s - theta_s;   // threshold_below: expr - theta

    assign block = (margin_q < 0) || (secded_status == 2'b10);
    assign margin_q_dbg = margin_q;
    assign theta_corrected_dbg = theta_corrected;
endmodule
