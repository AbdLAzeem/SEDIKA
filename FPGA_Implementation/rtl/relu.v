// =====================================================================
// relu.v
//
// Combinational signed ReLU: y = max(x, 0).
// One LUT-depth on Spartan 3E (sign bit selects between input and 0).
//
// Parameterised width so the same block can be reused on activations
// of different widths (e.g. after requantisation to INT16 or to a wider
// intermediate width).
// =====================================================================
`timescale 1ns / 1ps

module relu #(
    parameter integer W = 16
)(
    input  wire signed [W-1:0] x,
    output wire signed [W-1:0] y
);

    assign y = x[W-1] ? {W{1'b0}} : x;

endmodule
