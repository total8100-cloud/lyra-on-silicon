// H_i evaluation core for Lyra's Cyber-domain `zone_isolation` constraint
// (IEC 62443-3-3 SR 5.2): "unauthenticated lateral movement across security
// zones is blocked." Compiled form (auto_compiler.py, threshold_above):
//   expr = zone_distance * (1 - auth_state)
//   margin = theta - expr        ; BLOCK iff margin < 0
// Since auth_state in {0,1}, expr reduces to a mux (no multiplier needed) --
// this is the exact same arithmetic the Python compiler performs, just
// specialised for the binary auth_state the YAML schema documents.
//
// theta is protected by a SECDED-coded register (secded_decoder.v); an
// integrity failure (uncorrectable double-bit error) forces BLOCK regardless
// of the computed margin, in keeping with Lyra's fail-first philosophy
// (Paper 1 Sec 4.3's fallback_i mechanism).
`timescale 1ns/1ps
module lyra_zone_isolation_core (
    input  [15:0] zone_distance_q,   // Q4.12 unsigned fixed point, valid range [0,1)
    input         auth_state,        // 0=unauthenticated, 1=authenticated
    input  [21:0] theta_codeword,    // SECDED-protected compiled threshold (theta=0.30)
    output        block,             // 1 = BLOCK (violation or integrity failure)
    output [1:0]  secded_status,     // 00 ok, 01 corrected, 10 uncorrectable
    output [15:0] theta_corrected_dbg // diagnostic-only tap, inert for block/status
);
    wire [15:0] theta_corrected;

    secded_decoder #(.K(16), .M(5)) u_decoder (
        .codeword (theta_codeword),
        .data_out (theta_corrected),
        .status   (secded_status)
    );

    wire signed [16:0] expr_q  = auth_state ? 17'sd0 : {1'b0, zone_distance_q};
    wire signed [16:0] theta_s = {{1{theta_corrected[15]}}, theta_corrected};
    wire signed [16:0] margin_q = theta_s - expr_q;

    assign block = (margin_q < 0) || (secded_status == 2'b10);
    assign theta_corrected_dbg = theta_corrected;
endmodule
