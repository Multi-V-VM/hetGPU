module top(input logic clk, output logic [31:0] cnt);
logic [31:0] c = 0;
always_ff @(posedge clk) c <= c + 1;
assign cnt = c;
endmodule
