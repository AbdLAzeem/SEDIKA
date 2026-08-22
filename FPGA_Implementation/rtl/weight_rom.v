// =====================================================================
// weight_rom.v
//
// Synchronous BRAM-backed weight storage. One instance per layer.
// Initialised from a $readmemh memory file produced by quantize_ae.py.
//
// Storage layout (row-major):
//   Address = i * N_OUT + j
//   Word    = W_q[i, j]   (signed INT8)
//
// Where i is the input index (0..N_IN-1) and j is the output channel
// index (0..N_OUT-1). The top-level FSM presents one (input, channel)
// address pair per cycle; the ROM returns the corresponding signed
// INT8 weight on the next cycle (synchronous read).
//
// On Spartan 3E XC3S500E, each BlockRAM is 16 Kbits. A single 8x1024
// configuration easily fits any single layer of this AE (max layer is
// 25 * 25 * 8 bits = 5 Kbits for layer 4). Synthesis will infer a
// distributed-RAM implementation when DEPTH is small, which is often
// preferable for these tiny layers.
// =====================================================================
`timescale 1ns / 1ps

module weight_rom #(
    parameter integer DEPTH  = 400,             // total entries (e.g. N_IN * N_OUT)
    parameter integer ADDR_W = 9,               // ceil(log2(DEPTH))
    parameter integer DATA_W = 8,
    parameter MEM_FILE = "weights_L1.mem"        // path to $readmemh hex file
)(
    input  wire                  clk,
    input  wire                  en,
    input  wire [ADDR_W-1:0]     addr,
    output reg  signed [DATA_W-1:0] dout
);

    // Synthesis can map this to BRAM or distributed RAM at its discretion.
    (* RAM_STYLE = "block" *)
    reg [DATA_W-1:0] mem [0:DEPTH-1];

    initial begin
        $readmemh(MEM_FILE, mem);
    end

    always @(posedge clk) begin
        if (en) begin
            dout <= $signed(mem[addr]);
        end
    end

endmodule
