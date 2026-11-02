// Real-hardware fault-injection harness for Phase 2b-extension (PRD 9.6/9.8
// item 1). A Vivado VIO debug core (driven over JTAG by hw_server, scripted
// non-interactively by vivado/program_and_test.tcl) lets us XOR an arbitrary
// bit mask into the compiled theta codeword WHILE THE DESIGN IS RUNNING ON THE
// REAL ARTY S7 BOARD -- this is the "real hardware" counterpart to
// tb/tb_zone_isolation_secded.v's testbench-only fault injection, closing the
// "software/simulation predicts hardware" claim with an actual silicon
// measurement (PRD Sec 4.3's "2nd fault model" promoted from simulation to
// measurement).
`timescale 1ns/1ps
module lyra_zone_isolation_vio_top (
    input clk
);
    wire [21:0] fault_mask;
    wire [15:0] zone_distance_q;
    wire        auth_state;
    wire        block;
    wire [1:0]  secded_status;

    `include "zone_isolation_params.vh"

    wire [21:0] codeword_faulty = THETA_CODEWORD ^ fault_mask;

    lyra_zone_isolation_core u_core (
        .zone_distance_q (zone_distance_q),
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
        .probe_out1(zone_distance_q),
        .probe_out2(auth_state)
    );
endmodule
