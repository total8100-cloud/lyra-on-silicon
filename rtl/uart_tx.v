// Minimal free-running UART transmitter, 8N1. Used to bypass the JTAG/VIO
// debug-probe readback path (observed intermittently unreliable for the
// falling_glucose_guard constraint's wider signed probes) with a genuinely
// independent communication channel -- Arty S7's onboard FTDI USB-UART
// bridge, which shares no infrastructure with the JTAG debug core.
`timescale 1ns/1ps
module uart_tx #(
    parameter CLK_FREQ_HZ = 12_000_000,
    parameter BAUD        = 115_200
)(
    input        clk,
    input  [7:0] data,       // sampled live when a new byte starts
    output reg   txd
);
    localparam integer DIV = CLK_FREQ_HZ / BAUD;

    reg [15:0] clk_cnt = 0;
    reg [3:0]  bit_idx = 0;
    reg [9:0]  shift   = 10'b1111111111; // idle = all 1s (stop/idle line high)
    reg        busy    = 0;

    initial txd = 1'b1;

    always @(posedge clk) begin
        if (!busy) begin
            // idle: start a new frame with the CURRENT live data value
            shift   <= {1'b1, data, 1'b0}; // {stop, data[7:0], start}
            bit_idx <= 0;
            clk_cnt <= 0;
            busy    <= 1'b1;
            txd     <= 1'b0; // start bit immediately
        end else begin
            if (clk_cnt == DIV - 1) begin
                clk_cnt <= 0;
                if (bit_idx == 9) begin
                    busy <= 1'b0;
                    txd  <= 1'b1;
                end else begin
                    bit_idx <= bit_idx + 1;
                    txd     <= shift[bit_idx + 1];
                end
            end else begin
                clk_cnt <= clk_cnt + 1;
            end
        end
    end
endmodule
