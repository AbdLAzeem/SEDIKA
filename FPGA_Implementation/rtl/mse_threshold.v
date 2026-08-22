// =====================================================================
// mse_threshold.v
//
// Final-stage MSE accumulator + threshold comparator.
//
// Streams one (x, x_hat) pair per cycle while `enable` is high; on
// `done_in` (asserted in the cycle of the final pair), latches the
// final MSE-sum and produces the anomaly flag by comparing against
// the hard-coded threshold TAU_INT.
//
// Note: MSE-sum here is sum of squared INT16 differences (no /N).
// The threshold was pre-scaled by the calibration script
// (quantize_ae.py) to absorb the /N divide: tau_int = tau * N / S_in^2.
// For the int16/per-channel build TAU_INT = 32'd3784122 (decimal).
//
// Bit-widths:
//   ACT_W = 16  -> diff = signed 17-bit
//   diff_sq    = unsigned 33-bit (33 = 2*(ACT_W+1) - 1)
//   N = 25     -> sum needs +5 bits -> 38-bit unsigned accumulator
// =====================================================================
`timescale 1ns / 1ps

module mse_threshold #(
    parameter integer ACT_W   = 16,
    parameter integer N_FEAT  = 25,
    parameter integer SUM_W   = 38,
    parameter [31:0]  TAU_INT = 32'd3784122
)(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   start,         // pulse to clear accumulator
    input  wire                   enable,        // accumulate this cycle
    input  wire                   last,          // pulse with the final pair
    input  wire signed [ACT_W-1:0] x,
    input  wire signed [ACT_W-1:0] x_hat,
    output reg  [SUM_W-1:0]       mse_sum,       // running / final MSE-sum
    output reg                    valid,         // pulses when comparison ready
    output reg                    anomaly_flag   // (mse_sum > TAU_INT)
);

    wire signed [ACT_W:0]   diff    = x - x_hat;
    wire [2*(ACT_W+1)-1:0]  diff_sq = diff * diff;

    always @(posedge clk) begin
        if (!rst_n) begin
            mse_sum      <= {SUM_W{1'b0}};
            valid        <= 1'b0;
            anomaly_flag <= 1'b0;
        end else begin
            valid <= 1'b0;
            if (start) begin
                mse_sum      <= {SUM_W{1'b0}};
                anomaly_flag <= 1'b0;
            end else if (enable) begin
                mse_sum <= mse_sum + diff_sq;
            end
            if (last) begin
                // Comparison ready next cycle: use what mse_sum will become.
                anomaly_flag <= (mse_sum + diff_sq > TAU_INT) ? 1'b1 : 1'b0;
                valid        <= 1'b1;
            end
        end
    end

endmodule
