`timescale 1ns/1ps
// Self-checking testbench: software/hardware equivalence check for the
// SECDED-protected zone_isolation core (Project-10 PRD Sec 6.2 "software-
// hardware parity"). Reuses the exact theta codeword computed by
// pilot/lyra_secded_mitigation.py (via tools/gen_secded_params.py) and
// reproduces, in RTL simulation, the same claims validated in Python:
//   - no fault: BLOCK decision matches the true theta=0.30 rule
//   - any single codeword bit flip (all 22 bits): corrected (or harmless if
//     it was the p0 bit) -- BLOCK decision UNCHANGED from the no-fault case
//   - a double-bit flip: flagged uncorrectable -> BLOCK forced (fail-safe)
module tb_zone_isolation_secded;
    `include "zone_isolation_params.vh"

    reg  [15:0] zone_distance_q;
    reg         auth_state;
    reg  [21:0] codeword;
    wire        block;
    wire [1:0]  status;

    integer errors = 0;
    integer bit_i, bit_j;

    lyra_zone_isolation_core dut (
        .zone_distance_q (zone_distance_q),
        .auth_state      (auth_state),
        .theta_codeword  (codeword),
        .block           (block),
        .secded_status   (status)
    );

    // Q4.12 fixed point: 0.1 -> ~410, 0.5 -> 2048
    localparam [15:0] ZD_SAFE  = 16'd410;   // expr=0.1 < theta=0.30 -> SAFE
    localparam [15:0] ZD_BLOCK = 16'd2048;  // expr=0.5 > theta=0.30 -> BLOCK

    task check_state(input [15:0] zd, input expected_block, input [63:0] label);
        begin
            zone_distance_q = zd;
            auth_state = 1'b0;
            #1;
            if (block !== expected_block) begin
                $display("FAIL [%0s] zd=%0d codeword=%b -> block=%b status=%b (expected block=%b)",
                          label, zd, codeword, block, status, expected_block);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        // ---- 1. baseline (no fault) ----
        codeword = THETA_CODEWORD;
        check_state(ZD_SAFE,  1'b0, "baseline-SAFE");
        check_state(ZD_BLOCK, 1'b1, "baseline-BLOCK");
        if (status !== 2'b00) begin
            $display("FAIL baseline status should be 00 (ok), got %b", status);
            errors = errors + 1;
        end
        $display("Baseline OK: THETA_Q=%0d (0.30), ZD_SAFE->block=0, ZD_BLOCK->block=1", THETA_Q);

        // ---- 2. every single-bit fault in the 22-bit codeword ----
        for (bit_i = 0; bit_i < 22; bit_i = bit_i + 1) begin
            codeword = THETA_CODEWORD ^ (22'b1 << bit_i);
            check_state(ZD_SAFE,  1'b0, "single-bit-SAFE");
            check_state(ZD_BLOCK, 1'b1, "single-bit-BLOCK");
            // status must be 00 (p0-only error) or 01 (corrected) -- never 10
            if (status === 2'b10) begin
                $display("FAIL single-bit flip at bit %0d falsely reported uncorrectable", bit_i);
                errors = errors + 1;
            end
        end
        $display("Single-bit sweep complete: 22/22 bit positions tested x 2 states.");

        // ---- 3. double-bit faults: every pair of word-position bits (0..20, excludes p0 bit 21) ----
        for (bit_i = 0; bit_i < 21; bit_i = bit_i + 1) begin
            for (bit_j = bit_i + 1; bit_j < 21; bit_j = bit_j + 1) begin
                codeword = THETA_CODEWORD ^ (22'b1 << bit_i) ^ (22'b1 << bit_j);
                zone_distance_q = ZD_SAFE;  // the state that would otherwise be SAFE
                auth_state = 1'b0;
                #1;
                if (status !== 2'b10) begin
                    $display("FAIL double-bit flip (%0d,%0d) not flagged uncorrectable, status=%b",
                              bit_i, bit_j, status);
                    errors = errors + 1;
                end else if (block !== 1'b1) begin
                    $display("FAIL double-bit flip (%0d,%0d) flagged uncorrectable but BLOCK not forced",
                              bit_i, bit_j);
                    errors = errors + 1;
                end
            end
        end
        $display("Double-bit sweep complete: 210/210 word-bit pairs tested (all must force BLOCK).");

        if (errors == 0)
            $display("TB_ZONE_ISOLATION_SECDED PASS");
        else
            $display("TB_ZONE_ISOLATION_SECDED FAIL (%0d errors)", errors);

        $finish;
    end
endmodule
