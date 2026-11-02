// VIO harness around the REGISTERED lyra_zone_isolation_top, extended with a
// theta_corrected_dbg readback probe -- lets a host script collect the REAL,
// decoded-on-silicon theta value for a batch of triple-bit fault injections,
// to feed directly into the software-trained IsolationForest AI detector
// (pilot/lyra_ai_drift_detector.py) as a hardware-in-the-loop validation of
// the detector's intended deployment pipeline (PRD Sec 9.x AI-contribution
// strengthening pass).
`timescale 1ns/1ps
module lyra_zone_isolation_top_vio_aidbg (
    input clk
);
    wire [21:0] fault_mask;
    wire [15:0] zone_distance_q_raw;
    wire        auth_state;
    wire        block;
    wire [1:0]  secded_status;
    wire [15:0] theta_corrected_dbg;

    `include "zone_isolation_params.vh"

    wire [21:0] codeword_faulty = THETA_CODEWORD ^ fault_mask;

    lyra_zone_isolation_top u_top (
        .clk                 (clk),
        .zone_distance_q     (zone_distance_q_raw),
        .auth_state          (auth_state),
        .theta_codeword      (codeword_faulty),
        .block               (block),
        .secded_status       (secded_status),
        .theta_corrected_dbg (theta_corrected_dbg)
    );

    vio_lyra3 u_vio (
        .clk       (clk),
        .probe_in0 (block),
        .probe_in1 (secded_status),
        .probe_in2 (theta_corrected_dbg),
        .probe_out0(fault_mask),
        .probe_out1(zone_distance_q_raw),
        .probe_out2(auth_state)
    );
endmodule
