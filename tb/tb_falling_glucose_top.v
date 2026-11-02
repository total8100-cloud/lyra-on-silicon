`timescale 1ns/1ps
// Diagnostic: does the REGISTERED top-level pipeline (not yet exercised by
// any testbench) reproduce real-hardware mismatch seen on Arty S7?
module tb_falling_glucose_top;
    `include "falling_glucose_params.vh"

    reg clk = 0;
    reg signed [15:0] bg_rate_q;
    reg  [21:0] theta_codeword;
    wire        block;
    wire [1:0]  secded_status;

    always #5 clk = ~clk;

    lyra_falling_glucose_top dut (
        .clk(clk), .bg_rate_q(bg_rate_q), .theta_codeword(theta_codeword),
        .block(block), .secded_status(secded_status)
    );

    initial begin
        theta_codeword = THETA_CODEWORD;
        bg_rate_q = 16'sd0;
        repeat (4) @(posedge clk);
        $display("t=%0t SAFE: bg=%0d block=%b status=%b", $time, bg_rate_q, block, secded_status);

        bg_rate_q = -16'sd2458;
        repeat (4) @(posedge clk);
        $display("t=%0t BLOCK: bg=%0d block=%b status=%b", $time, bg_rate_q, block, secded_status);

        $finish;
    end
endmodule
