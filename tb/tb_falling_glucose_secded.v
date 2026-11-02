`timescale 1ns/1ps
// Self-checking testbench for the SECOND hardware-validated constraint
// (Medical/falling_glucose_guard, theta=-0.30, threshold_below). Mirrors
// tb_zone_isolation_secded.v exactly: baseline, exhaustive single-bit
// (22/22) and exhaustive double-bit (210/210) codeword fault sweep.
module tb_falling_glucose_secded;
    `include "falling_glucose_params.vh"

    reg  signed [15:0] bg_rate_q;
    reg  [21:0] codeword;
    wire        block;
    wire [1:0]  status;

    integer errors = 0;
    integer bit_i, bit_j;

    lyra_falling_glucose_core dut (
        .bg_rate_q      (bg_rate_q),
        .theta_codeword (codeword),
        .block          (block),
        .secded_status  (status)
    );

    // Q4.12: 0.0 -> safe (expr=0.0 >= theta=-0.30), -0.60 -> block (expr=-0.60 < -0.30)
    localparam signed [15:0] BG_SAFE  = 16'sd0;
    localparam signed [15:0] BG_BLOCK = -16'sd2458;

    task check_state(input signed [15:0] bg, input expected_block, input [63:0] label);
        begin
            bg_rate_q = bg;
            #1;
            if (block !== expected_block) begin
                $display("FAIL [%0s] bg=%0d codeword=%b -> block=%b status=%b (expected block=%b)",
                          label, bg, codeword, block, status, expected_block);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        codeword = THETA_CODEWORD;
        check_state(BG_SAFE,  1'b0, "baseline-SAFE");
        check_state(BG_BLOCK, 1'b1, "baseline-BLOCK");
        if (status !== 2'b00) begin
            $display("FAIL baseline status should be 00 (ok), got %b", status);
            errors = errors + 1;
        end
        $display("Baseline OK: THETA_Q=%0d (-0.30), BG_SAFE->block=0, BG_BLOCK->block=1", THETA_Q);

        for (bit_i = 0; bit_i < 22; bit_i = bit_i + 1) begin
            codeword = THETA_CODEWORD ^ (22'b1 << bit_i);
            check_state(BG_SAFE,  1'b0, "single-bit-SAFE");
            check_state(BG_BLOCK, 1'b1, "single-bit-BLOCK");
            if (status === 2'b10) begin
                $display("FAIL single-bit flip at bit %0d falsely reported uncorrectable", bit_i);
                errors = errors + 1;
            end
        end
        $display("Single-bit sweep complete: 22/22 bit positions tested x 2 states.");

        for (bit_i = 0; bit_i < 21; bit_i = bit_i + 1) begin
            for (bit_j = bit_i + 1; bit_j < 21; bit_j = bit_j + 1) begin
                codeword = THETA_CODEWORD ^ (22'b1 << bit_i) ^ (22'b1 << bit_j);
                bg_rate_q = BG_SAFE;
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
            $display("TB_FALLING_GLUCOSE_SECDED PASS");
        else
            $display("TB_FALLING_GLUCOSE_SECDED FAIL (%0d errors)", errors);

        $finish;
    end
endmodule
