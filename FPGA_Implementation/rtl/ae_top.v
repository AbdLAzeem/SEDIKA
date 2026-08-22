// =====================================================================
// ae_top.v
//
// Top-level FSM for the SEDIKA Tier-3 autoencoder inference engine.
// Architecture: 25 -> 16 -> 8 -> 16 -> 25 dense layers, ReLU on layers
// 1-3, linear on layer 4. Input/output activations are signed INT16,
// weights signed INT8, biases signed INT32, per-channel requantisation.
//
// Resource budget on Spartan 3E XC3S500E:
//   - 20 dedicated 18x18 multipliers (one per parallel MAC unit)
//   - One MAC per output channel of the currently-active layer
//   - Time-multiplex: layer 4 (25 outputs) runs in 2 sub-passes
//     using 20 + 5 MAC units.
//
// Compute schedule:
//   Layer 1 (25 in, 16 out):  25 input cycles + finalize
//   Layer 2 (16 in,  8 out):  16 input cycles + finalize
//   Layer 3 (8  in, 16 out):   8 input cycles + finalize
//   Layer 4a (16 in, 20 out): 16 input cycles + finalize  (channels 0..19)
//   Layer 4b (16 in,  5 out): 16 input cycles + finalize  (channels 20..24)
//
// At 50 MHz, MAC-only latency is ~81 cycles ~= 1.6 us; total inference
// including state-machine overhead and MSE estimated at ~3-5 us.
//
// Interface:
//   start             pulse high for one cycle when 25-vector x_in is valid
//   x_flat            25 INT16 inputs packed LSB-first, signed: total 400 bits
//   anomaly_flag      asserted when MSE-sum > tau_int (raw, no /N)
//   mse_sum_out       33-bit unsigned MSE-sum (for debug / waveform capture)
//   done              pulses high for one cycle when inference completes
// =====================================================================
`timescale 1ns / 1ps

module ae_top #(
    parameter integer ACT_W = 16,
    parameter integer W_W   = 8,
    parameter integer ACC_W = 40,
    parameter integer BIAS_W = 32,
    parameter integer N_FEAT = 25,
    // Pre-scaled threshold (sum of diff^2, no /N).
    // For the int16/per-channel build the value is 3_784_122 (decimal).
    parameter [31:0] TAU_INT = 32'd3784122
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  start,
    input  wire signed [N_FEAT*ACT_W-1:0] x_flat,
    output reg                   done,
    output reg                   anomaly_flag,
    output reg  [32:0]           mse_sum_out
);

    // -----------------------------------------------------------------
    // Convenience aliases for input vector elements
    // -----------------------------------------------------------------
    wire signed [ACT_W-1:0] x_in [0:N_FEAT-1];
    genvar gi;
    generate
        for (gi = 0; gi < N_FEAT; gi = gi + 1) begin : g_xin
            assign x_in[gi] = x_flat[(gi+1)*ACT_W-1 -: ACT_W];
        end
    endgenerate

    // -----------------------------------------------------------------
    // Activation buffers (ping-pong)
    // -----------------------------------------------------------------
    // Max width any buffer needs: layer 1 input (25) and layer 4 output (25).
    reg signed [ACT_W-1:0] act_buf_a [0:N_FEAT-1];
    reg signed [ACT_W-1:0] act_buf_b [0:N_FEAT-1];
    reg                    buf_sel;       // 0: read A write B, 1: read B write A

    // -----------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------
    localparam S_IDLE       = 5'd0;
    localparam S_LOAD       = 5'd1;
    localparam S_L1_MAC     = 5'd2;
    localparam S_L1_FIN     = 5'd3;
    localparam S_L2_MAC     = 5'd4;
    localparam S_L2_FIN     = 5'd5;
    localparam S_L3_MAC     = 5'd6;
    localparam S_L3_FIN     = 5'd7;
    localparam S_L4A_MAC    = 5'd8;
    localparam S_L4A_FIN    = 5'd9;
    localparam S_L4B_MAC    = 5'd10;
    localparam S_L4B_FIN    = 5'd11;
    localparam S_MSE        = 5'd12;
    localparam S_DONE       = 5'd13;

    reg [4:0] state, state_n;

    // Per-state cycle counter
    reg [5:0] cycle;

    // -----------------------------------------------------------------
    // 20 parallel MAC units (max channels processed simultaneously).
    // Layer 4 uses 20 in the first sub-pass and 5 in the second sub-pass.
    // Smaller layers use only the first N MACs.
    // -----------------------------------------------------------------
    localparam integer N_MAC = 20;

    reg                            mac_clear;
    reg                            mac_enable;
    reg  signed [ACT_W-1:0]        mac_act_bcast;
    wire signed [W_W-1:0]          mac_weight  [0:N_MAC-1];
    wire signed [ACC_W-1:0]        mac_acc     [0:N_MAC-1];

    genvar gm;
    generate
        for (gm = 0; gm < N_MAC; gm = gm + 1) begin : g_mac
            mac_unit #(.ACT_W(ACT_W), .W_W(W_W), .ACC_W(ACC_W)) u_mac (
                .clk       (clk),
                .rst_n     (rst_n),
                .clear     (mac_clear),
                .enable    (mac_enable),
                .act_in    (mac_act_bcast),
                .weight_in (mac_weight[gm]),
                .acc       (mac_acc[gm])
            );
        end
    endgenerate

    // -----------------------------------------------------------------
    // Weight ROMs: one per layer. Each layer's ROM is read via a single
    // address bus; the data bus is wide (N_MAC * 8 bits) so 20 weights
    // for the current input row are fetched in parallel.
    // For a real Spartan 3E implementation, the ROM is decomposed into
    // N_MAC small distributed-RAM slices populated row-major (input
    // index outer, channel index inner); here we model the same layout
    // with a packed wide ROM.
    // -----------------------------------------------------------------
    // Layer dimensions
    localparam integer L1_IN = 25, L1_OUT = 16;
    localparam integer L2_IN = 16, L2_OUT = 8;
    localparam integer L3_IN = 8,  L3_OUT = 16;
    localparam integer L4_IN = 16, L4_OUT = 25;

    // Weight memories. We use direct register arrays initialised at
    // synthesis time via $readmemh — Vivado/ISE will infer BRAM/dist-RAM.
    reg signed [W_W-1:0] W1 [0:L1_IN*L1_OUT-1];   // 400 bytes
    reg signed [W_W-1:0] W2 [0:L2_IN*L2_OUT-1];   // 128
    reg signed [W_W-1:0] W3 [0:L3_IN*L3_OUT-1];   // 128
    reg signed [W_W-1:0] W4 [0:L4_IN*L4_OUT-1];   // 400

    // Biases (INT32, per channel)
    reg signed [BIAS_W-1:0] B1 [0:L1_OUT-1];
    reg signed [BIAS_W-1:0] B2 [0:L2_OUT-1];
    reg signed [BIAS_W-1:0] B3 [0:L3_OUT-1];
    reg signed [BIAS_W-1:0] B4 [0:L4_OUT-1];

    // Per-channel requant: packed 32-bit word per channel
    //   bits [31:16] = m_int (unsigned 16-bit multiplier)
    //   bits [ 4: 0] = shift  (right-shift count, max 31)
    reg [31:0] RM1 [0:L1_OUT-1];
    reg [31:0] RM2 [0:L2_OUT-1];
    reg [31:0] RM3 [0:L3_OUT-1];
    reg [31:0] RM4 [0:L4_OUT-1];

    wire [15:0] RM1_M [0:L1_OUT-1];
    wire [4:0]  RM1_S [0:L1_OUT-1];
    wire [15:0] RM2_M [0:L2_OUT-1];
    wire [4:0]  RM2_S [0:L2_OUT-1];
    wire [15:0] RM3_M [0:L3_OUT-1];
    wire [4:0]  RM3_S [0:L3_OUT-1];
    wire [15:0] RM4_M [0:L4_OUT-1];
    wire [4:0]  RM4_S [0:L4_OUT-1];

    genvar gu;
    generate
        for (gu = 0; gu < L1_OUT; gu = gu + 1) begin : g_rm1
            assign RM1_M[gu] = RM1[gu][31:16];
            assign RM1_S[gu] = RM1[gu][ 4: 0];
        end
        for (gu = 0; gu < L2_OUT; gu = gu + 1) begin : g_rm2
            assign RM2_M[gu] = RM2[gu][31:16];
            assign RM2_S[gu] = RM2[gu][ 4: 0];
        end
        for (gu = 0; gu < L3_OUT; gu = gu + 1) begin : g_rm3
            assign RM3_M[gu] = RM3[gu][31:16];
            assign RM3_S[gu] = RM3[gu][ 4: 0];
        end
        for (gu = 0; gu < L4_OUT; gu = gu + 1) begin : g_rm4
            assign RM4_M[gu] = RM4[gu][31:16];
            assign RM4_S[gu] = RM4[gu][ 4: 0];
        end
    endgenerate

    initial begin
        $readmemh("weights_L1.mem", W1);
        $readmemh("weights_L2.mem", W2);
        $readmemh("weights_L3.mem", W3);
        $readmemh("weights_L4.mem", W4);
        $readmemh("biases_L1.mem",  B1);
        $readmemh("biases_L2.mem",  B2);
        $readmemh("biases_L3.mem",  B3);
        $readmemh("biases_L4.mem",  B4);
        $readmemh("requant_L1.mem", RM1);
        $readmemh("requant_L2.mem", RM2);
        $readmemh("requant_L3.mem", RM3);
        $readmemh("requant_L4.mem", RM4);
    end

    // -----------------------------------------------------------------
    // Layer dispatcher: based on state, route weight + bias + input to MACs
    // -----------------------------------------------------------------
    integer mac_idx;
    integer ch_offset;   // for L4 sub-passes
    reg signed [W_W-1:0] mac_weight_r [0:N_MAC-1];
    assign mac_weight = mac_weight_r;

    always @(*) begin
        // Default: zero out all MAC weights to keep tools happy on
        // unused MAC channels for smaller layers.
        for (mac_idx = 0; mac_idx < N_MAC; mac_idx = mac_idx + 1)
            mac_weight_r[mac_idx] = {W_W{1'b0}};

        case (state)
            S_L1_MAC: begin
                for (mac_idx = 0; mac_idx < L1_OUT; mac_idx = mac_idx + 1)
                    mac_weight_r[mac_idx] = W1[cycle*L1_OUT + mac_idx];
            end
            S_L2_MAC: begin
                for (mac_idx = 0; mac_idx < L2_OUT; mac_idx = mac_idx + 1)
                    mac_weight_r[mac_idx] = W2[cycle*L2_OUT + mac_idx];
            end
            S_L3_MAC: begin
                for (mac_idx = 0; mac_idx < L3_OUT; mac_idx = mac_idx + 1)
                    mac_weight_r[mac_idx] = W3[cycle*L3_OUT + mac_idx];
            end
            S_L4A_MAC: begin
                // channels 0..19 (full N_MAC active)
                for (mac_idx = 0; mac_idx < N_MAC; mac_idx = mac_idx + 1)
                    mac_weight_r[mac_idx] = W4[cycle*L4_OUT + mac_idx];
            end
            S_L4B_MAC: begin
                // channels 20..24 (only first 5 active)
                for (mac_idx = 0; mac_idx < 5; mac_idx = mac_idx + 1)
                    mac_weight_r[mac_idx] = W4[cycle*L4_OUT + (20 + mac_idx)];
            end
            default: ;  // weights stay zeroed
        endcase
    end

    // -----------------------------------------------------------------
    // Input broadcast: select which activation feeds the MAC array
    // -----------------------------------------------------------------
    always @(*) begin
        case (state)
            S_L1_MAC:  mac_act_bcast = act_buf_a[cycle];   // input vector
            S_L2_MAC:  mac_act_bcast = act_buf_b[cycle];   // layer 1 output
            S_L3_MAC:  mac_act_bcast = act_buf_a[cycle];   // layer 2 output
            S_L4A_MAC: mac_act_bcast = act_buf_b[cycle];   // layer 3 output
            S_L4B_MAC: mac_act_bcast = act_buf_b[cycle];   // same source, 2nd sub-pass
            default:   mac_act_bcast = {ACT_W{1'b0}};
        endcase
    end

    // -----------------------------------------------------------------
    // Requantisation helper (combinational): acc -> INT16
    // y = ((acc + bias) * m + (1<<(shift-1))) >>> shift, saturated to INT16
    // -----------------------------------------------------------------
    function signed [ACT_W-1:0] requantize;
        input signed [ACC_W-1:0] acc_in;
        input signed [BIAS_W-1:0] bias_in;
        input [15:0] m;
        input [4:0]  sh;
        input apply_relu;
        reg signed [ACC_W:0]   acc_b;   // acc + bias
        reg signed [ACC_W+16:0] prod;
        reg signed [ACC_W+16:0] rounded;
        reg signed [ACC_W+16:0] shifted;
        integer i;
        begin
            acc_b = acc_in + bias_in;
            prod = acc_b * $signed({1'b0, m});
            rounded = prod + ((sh > 0) ? ($signed(1'b1) <<< (sh - 1)) : 0);
            shifted = rounded >>> sh;
            // Apply ReLU (if requested) and saturate.
            if (apply_relu && shifted[ACC_W+16])  // negative
                requantize = {ACT_W{1'b0}};
            else if (shifted > $signed({16'h0, 16'h7FFF}))
                requantize = 16'h7FFF;
            else if (shifted < $signed(-32768))
                requantize = -16'sh8000;
            else
                requantize = shifted[ACT_W-1:0];
        end
    endfunction

    // -----------------------------------------------------------------
    // MSE accumulator and threshold compare
    // -----------------------------------------------------------------
    reg [32:0] mse_sum;          // 33-bit unsigned
    reg [5:0]  mse_idx;
    wire signed [ACT_W:0] mse_diff = $signed({x_in[mse_idx][ACT_W-1], x_in[mse_idx]})
                                    - $signed({act_buf_a[mse_idx][ACT_W-1], act_buf_a[mse_idx]});
    wire [32:0] mse_diff_sq = mse_diff * mse_diff;  // 34-bit unsigned-equivalent

    // -----------------------------------------------------------------
    // Main FSM
    // -----------------------------------------------------------------
    integer k;
    always @(posedge clk) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            cycle        <= 6'd0;
            done         <= 1'b0;
            anomaly_flag <= 1'b0;
            mse_sum      <= 33'd0;
            mse_idx      <= 6'd0;
            mse_sum_out  <= 33'd0;
            mac_clear    <= 1'b1;
            mac_enable   <= 1'b0;
            buf_sel      <= 1'b0;
        end else begin
            done       <= 1'b0;
            mac_clear  <= 1'b0;
            mac_enable <= 1'b0;

            case (state)
                // -----------------------------------------------------
                S_IDLE: if (start) begin
                    // Latch x_in into act_buf_a (the "input" buffer).
                    for (k = 0; k < N_FEAT; k = k + 1) act_buf_a[k] <= x_in[k];
                    mse_sum   <= 33'd0;
                    cycle     <= 6'd0;
                    mac_clear <= 1'b1;
                    state     <= S_L1_MAC;
                end
                // -----------------------------------------------------
                // Layer 1: 25 input cycles, 16 MAC channels active.
                S_L1_MAC: begin
                    mac_enable <= 1'b1;
                    cycle      <= cycle + 1;
                    if (cycle == L1_IN - 1) begin
                        cycle <= 6'd0;
                        state <= S_L1_FIN;
                    end
                end
                S_L1_FIN: begin
                    // Apply bias + requantisation + ReLU per channel; store to act_buf_b
                    for (k = 0; k < L1_OUT; k = k + 1)
                        act_buf_b[k] <= requantize(mac_acc[k], B1[k], RM1_M[k], RM1_S[k], 1'b1);
                    mac_clear <= 1'b1;
                    cycle <= 6'd0;
                    state <= S_L2_MAC;
                end
                // -----------------------------------------------------
                S_L2_MAC: begin
                    mac_enable <= 1'b1;
                    cycle <= cycle + 1;
                    if (cycle == L2_IN - 1) begin cycle <= 0; state <= S_L2_FIN; end
                end
                S_L2_FIN: begin
                    for (k = 0; k < L2_OUT; k = k + 1)
                        act_buf_a[k] <= requantize(mac_acc[k], B2[k], RM2_M[k], RM2_S[k], 1'b1);
                    mac_clear <= 1'b1;
                    cycle <= 0;
                    state <= S_L3_MAC;
                end
                // -----------------------------------------------------
                S_L3_MAC: begin
                    mac_enable <= 1'b1;
                    cycle <= cycle + 1;
                    if (cycle == L3_IN - 1) begin cycle <= 0; state <= S_L3_FIN; end
                end
                S_L3_FIN: begin
                    for (k = 0; k < L3_OUT; k = k + 1)
                        act_buf_b[k] <= requantize(mac_acc[k], B3[k], RM3_M[k], RM3_S[k], 1'b1);
                    mac_clear <= 1'b1;
                    cycle <= 0;
                    state <= S_L4A_MAC;
                end
                // -----------------------------------------------------
                // Layer 4 sub-pass A: channels 0..19
                S_L4A_MAC: begin
                    mac_enable <= 1'b1;
                    cycle <= cycle + 1;
                    if (cycle == L4_IN - 1) begin cycle <= 0; state <= S_L4A_FIN; end
                end
                S_L4A_FIN: begin
                    for (k = 0; k < N_MAC; k = k + 1)
                        act_buf_a[k] <= requantize(mac_acc[k], B4[k], RM4_M[k], RM4_S[k], 1'b0);
                    mac_clear <= 1'b1;
                    cycle <= 0;
                    state <= S_L4B_MAC;
                end
                // Layer 4 sub-pass B: channels 20..24
                S_L4B_MAC: begin
                    mac_enable <= 1'b1;
                    cycle <= cycle + 1;
                    if (cycle == L4_IN - 1) begin cycle <= 0; state <= S_L4B_FIN; end
                end
                S_L4B_FIN: begin
                    for (k = 0; k < 5; k = k + 1)
                        act_buf_a[20 + k] <= requantize(mac_acc[k], B4[20 + k], RM4_M[20 + k], RM4_S[20 + k], 1'b0);
                    mse_idx <= 0;
                    mse_sum <= 0;
                    state <= S_MSE;
                end
                // -----------------------------------------------------
                // MSE: sum of (x_in[i] - x_hat[i])^2 over all 25 features.
                // x_hat is now in act_buf_a; x_in is read from the latched x_flat.
                S_MSE: begin
                    mse_sum <= mse_sum + mse_diff_sq;
                    if (mse_idx == N_FEAT - 1) begin
                        state <= S_DONE;
                    end
                    mse_idx <= mse_idx + 1;
                end
                // -----------------------------------------------------
                S_DONE: begin
                    mse_sum_out  <= mse_sum;
                    anomaly_flag <= (mse_sum > TAU_INT) ? 1'b1 : 1'b0;
                    done         <= 1'b1;
                    state        <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
