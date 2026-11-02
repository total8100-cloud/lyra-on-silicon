// VIO harness around the REGISTERED lyra_zone_isolation_top (the exact
// netlist whose LUT 66 / FF 42 / Fmax~200MHz numbers were reported in
// Project-10 PRD Sec 9.6) -- closes the gap flagged 2026-10-15: that build
// was placed-and-routed for the real xc7s50 part but never turned into a
// bitstream or run on the chip; Sec 9.7's live hardware test exercised
// lyra_zone_isolation_core directly (functionally identical logic) but not
// the registered pipeline stage this exact resource count depends on.
`timescale 1ns/1ps
module lyra_zone_isolation_top_vio (
    input clk
);
    wire [21:0] fault_mask;
    wire [15:0] zone_distance_q_raw;
    wire        auth_state;
    wire        block;
    wire [1:0]  secded_status;

    `include "zone_isolation_params.vh"

    wire [21:0] codeword_faulty = THETA_CODEWORD ^ fault_mask;

    lyra_zone_isolation_top u_top (
        .clk             (clk),
        .zone_distance_q (zone_distance_q_raw),
        .auth_state      (auth_state),
        .theta_codeword  (codeword_faulty),
        .block           (block),
        .secded_status   (secded_status)
    );

    vio_lyra u_vio (
        .clk       (clk),
        .probe_in0 (block),
        .probe_in1 (secded_status),
        .probe_out0(fault_mask),
        .probe_out1(zone_distance_q_raw),
        .probe_out2(auth_state)
    );
endmodule
