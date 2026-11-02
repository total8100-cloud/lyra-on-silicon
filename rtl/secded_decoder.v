// Generic Hamming SECDED decoder (Single Error Correct, Double Error Detect).
// K data bits + M Hamming parity bits (at power-of-two positions, 1-indexed)
// + 1 overall parity bit for double-error detection. Matches the scheme
// validated in pilot/lyra_secded_mitigation.py and the codeword layout
// produced by tools/gen_secded_params.py:
//   codeword[K+M-1:0] = Hamming word positions 1..(K+M), codeword[i-1]=word[i]
//   codeword[K+M]     = overall parity p0
`timescale 1ns/1ps
module secded_decoder #(
    parameter K = 16,
    parameter M = 5
)(
    input  [K+M:0]   codeword,
    output reg [K-1:0] data_out,
    output reg [1:0]   status        // 2'b00=ok, 2'b01=corrected, 2'b10=uncorrectable
);
    localparam N = K + M;

    integer i;
    integer data_idx;
    reg [M:0] syndrome;
    reg       overall;
    reg [N:1] word_in;
    reg [N:1] word_c;
    reg       p0_in;

    always @* begin
        for (i = 1; i <= N; i = i + 1)
            word_in[i] = codeword[i-1];
        p0_in = codeword[N];

        syndrome = {(M+1){1'b0}};
        for (i = 1; i <= N; i = i + 1)
            if (word_in[i]) syndrome = syndrome ^ i[M:0];

        overall = p0_in;
        for (i = 1; i <= N; i = i + 1)
            overall = overall ^ word_in[i];

        word_c = word_in;

        if (syndrome == 0 && overall == 1'b0) begin
            status = 2'b00;                       // no error
        end else if (syndrome != 0 && overall == 1'b1) begin
            word_c[syndrome] = ~word_in[syndrome]; // single-bit error -> corrected
            status = 2'b01;
        end else if (syndrome != 0 && overall == 1'b0) begin
            status = 2'b10;                        // double-bit error -> uncorrectable
        end else begin
            status = 2'b00;                        // error in p0 itself; data intact
        end

        data_idx = 0;
        for (i = 1; i <= N; i = i + 1) begin
            if ((i & (i - 1)) != 0) begin           // non-power-of-two -> data position
                data_out[data_idx] = word_c[i];
                data_idx = data_idx + 1;
            end
        end
    end
endmodule
