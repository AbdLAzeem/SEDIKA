// =====================================================================
// mac_unit.v
//
// Signed multiply-accumulate unit. One MAC per output channel of the
// currently-active layer. The top-level FSM broadcasts one input value
// across all MACs each cycle; each MAC fetches its corresponding weight
// from its assigned slice of the weight ROM.
//
// Parameters:
//   ACT_W   activation width (signed)
//   W_W     weight width (signed)
//   ACC_W   accumulator width (signed)
//
// Operation:
//   on `clear`:  acc <- 0
//   on `enable`: acc <- acc + (act_in * weight_in)
//
// Synthesises to one 18x18 DSP48-style multiplier on Spartan 3E. With
// ACT_W=16 and W_W=8, the multiplication fits comfortably inside the
// Spartan 3E XC3S500E's 18x18 dedicated multipliers (one per MAC unit).
// =====================================================================
`timescale 1ns / 1ps

module mac_unit #(
    parameter integer ACT_W = 16,
    parameter integer W_W   = 8,
    parameter integer ACC_W = 40
)(
    input  wire                  clk,
    input  wire                  rst_n,        // synchronous, active-low
    input  wire                  clear,        // synchronous clear of acc
    input  wire                  enable,       // accumulate this cycle
    input  wire signed [ACT_W-1:0]  act_in,
    input  wire signed [W_W-1:0]    weight_in,
    output reg  signed [ACC_W-1:0]  acc
);

    // Product is sign-extended for the accumulator.
    wire signed [ACT_W+W_W-1:0] product = act_in * weight_in;

    always @(posedge clk) begin
        if (!rst_n) begin
            acc <= {ACC_W{1'b0}};
        end else if (clear) begin
            acc <= {ACC_W{1'b0}};
        end else if (enable) begin
            // Sign-extend product to ACC_W before adding.
            acc <= acc + {{(ACC_W-(ACT_W+W_W)){product[ACT_W+W_W-1]}}, product};
        end
    end

endmodule
