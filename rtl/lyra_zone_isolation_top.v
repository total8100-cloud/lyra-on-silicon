// Synchronous wrapper around lyra_zone_isolation_core for Phase 2b resource/
// timing characterisation on the real Arty S7 (xc7s50csga324-1). Registers
// inputs and outputs at a single clock edge, matching the "evaluate once per
// decision step" model Lyra actually uses (Paper 1 Fig. 1 decision cycle) --
// not a purely combinational probe.
`timescale 1ns/1ps
module lyra_zone_isolation_top (
    input         clk,
    input  [15:0] zone_distance_q,
    input         auth_state,
    input  [21:0] theta_codeword,
    output reg    block,
    output reg [1:0] secded_status,
    output reg [15:0] theta_corrected_dbg
);
    reg  [15:0] zd_r;
    reg         auth_r;
    reg  [21:0] cw_r;
    wire        block_c;
    wire [1:0]  status_c;
    wire [15:0] theta_c;

    always @(posedge clk) begin
        zd_r   <= zone_distance_q;
        auth_r <= auth_state;
        cw_r   <= theta_codeword;
    end

    lyra_zone_isolation_core u_core (
        .zone_distance_q    (zd_r),
        .auth_state         (auth_r),
        .theta_codeword     (cw_r),
        .block              (block_c),
        .secded_status      (status_c),
        .theta_corrected_dbg(theta_c)
    );

    always @(posedge clk) begin
        block               <= block_c;
        secded_status       <= status_c;
        theta_corrected_dbg <= theta_c;
    end
endmodule
