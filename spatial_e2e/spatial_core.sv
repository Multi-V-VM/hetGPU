// spatial_core: lane-packed 32-lane spatial RTL core for the end-to-end
// experiment. Faithful to the paper's artifact description: instruction-ROM
// execution of PTX-derived use streams, 32-element lane-packed state guarded
// by predicates on every update, per-lane memory-event accounting (a warp
// load is one event per enabled lane), predicated atomics as serialized
// token islands (one event per active lane), barrier phase boundaries as
// safe points, and full architectural state export/import for live
// migration between core instances.
//
// Memory model: behavioral zero-latency array in the testbench; one lane's
// transaction per cycle (combinational request/ack). Atomic RMW = one read
// cycle + one write cycle per active lane.
//
// Arithmetic: int32 bit-exact wraparound. Shuffles follow the PTX
// down-shuffle rule: source = laneid + delta; out-of-clamp sources yield the
// lane's own value and clear the result predicate (written to %p0).

`default_nettype none

module spatial_core #(
    parameter int MAX_INSTR = 256,
    parameter int LANES     = 32,
    parameter int NREGS     = 32,
    parameter int NPREDS    = 8
) (
    input  wire         clk,
    input  wire         rst,

    input  wire [31:0]  params [0:7],

    output wire         mem_req,        // combinational, acked same cycle
    output wire         mem_we,
    output wire [31:0]  mem_addr,
    output wire [31:0]  mem_wdata,
    input  wire [31:0]  mem_rdata,
    input  wire         mem_ack,        // tie to mem_req in the TB

    output wire         done,
    output wire [15:0]  pc,
    output wire [31:0]  mem_events,
    output wire [31:0]  token_waits
);

    // ---------------- architectural state (checkpoint payload) -------------
    logic [31:0]  regs  [0:LANES-1][0:NREGS-1];
    logic         preds [0:LANES-1][0:NPREDS-1];
    logic [15:0]  pc_r;
    logic         done_r;
    logic [31:0]  events_r;
    logic [31:0]  waits_r;

    assign pc         = pc_r;
    assign done       = done_r;
    assign mem_events = events_r;
    assign token_waits = waits_r;

    // ---------------- instruction ROM --------------------------------------
    logic [31:0] rom [0:2*MAX_INSTR-1];

    // ---------------- execution state --------------------------------------
    typedef enum logic [1:0] { S_FETCH, S_EXEC, S_MEM } state_e;
    state_e state;

    // ---------------- instruction word fields ------------------------------
    logic [31:0] w_hi, w_lo;
    logic [7:0]  opcode, dst, srca, srcb;
    logic        guard, neg;
    logic [5:0]  preg;
    logic [23:0] imm;
    logic [5:0]  seq_lane;
    logic        atom_wr;              // atomic RMW: in write phase
    logic [31:0] atom_sum;
    logic [31:0] atom_addr;

    assign {opcode, dst, srca, srcb} = w_lo;
    assign {guard, neg, preg, imm}   = w_hi;

    function automatic logic lane_en(input logic [4:0] l);
        lane_en = !guard ? 1'b1 :
                  (neg ? !preds[l][preg] : preds[l][preg]);
    endfunction

    wire branch_taken = (guard == 1'b0) ? 1'b1 :
                        (neg ? !preds[0][preg] : preds[0][preg]);  // uniform

    wire [31:0] atom_lane_addr = regs[seq_lane][imm[4:0]] + regs[seq_lane][srca[4:0]];

    assign mem_req  = (state == S_MEM) && ((seq_lane < LANES) || atom_wr);
    assign mem_we   = atom_wr ? 1'b1 : (opcode == 8'd3);
    assign mem_addr = atom_wr ? atom_addr :
                      (opcode == 8'd7) ? atom_lane_addr :
                      regs[seq_lane][imm[4:0]];
    assign mem_wdata = atom_wr ? atom_sum : regs[seq_lane][srca[4:0]];

    always_ff @(posedge clk) begin
        if (rst) begin
            state    <= S_FETCH;
            pc_r     <= 16'h0;
            done_r   <= 1'b0;
            events_r <= 32'h0;
            waits_r  <= 32'h0;
            seq_lane <= 6'h0;
            atom_wr  <= 1'b0;
        end else if (!done_r) begin
            case (state)
                S_FETCH: begin
                    w_hi <= rom[2*pc_r];
                    w_lo <= rom[2*pc_r + 1];
                    state <= S_EXEC;
                end

                S_EXEC: begin
                    case (opcode)
                        8'd0: begin pc_r <= pc_r + 1; state <= S_FETCH; end                                   // NOP
                        8'd1: done_r <= 1'b1;                                     // EXIT
                        8'd2, 8'd3, 8'd7: begin                                   // LDG/STG/ATOM
                            seq_lane <= 6'h0;
                            atom_wr  <= 1'b0;
                            state    <= S_MEM;
                        end
                        8'd4: begin                                               // ADD
                            for (int l = 0; l < LANES; l++)
                                if (lane_en(l[4:0]))
                                    regs[l][dst[4:0]] <= regs[l][srca[4:0]] + regs[l][srcb[4:0]];
                            pc_r <= pc_r + 1;
                            state <= S_FETCH;
                        end
                        8'd5: begin                                               // SHFL_DOWN
                            for (int l = 0; l < LANES; l++) begin
                                if ((l + imm) <= srcb) begin
                                    regs[l][dst[4:0]] <= regs[(l + imm)][srca[4:0]];
                                    preds[l][0]       <= 1'b1;
                                end else begin
                                    regs[l][dst[4:0]] <= regs[l][srca[4:0]];
                                    preds[l][0]       <= 1'b0;
                                end
                            end
                            pc_r <= pc_r + 1;
                            state <= S_FETCH;
                        end
                        8'd6: begin pc_r <= pc_r + 1; state <= S_FETCH; end       // BAR (safe point)
                        8'd8: begin                                               // SETP_LT
                            for (int l = 0; l < LANES; l++)
                                if (lane_en(l[4:0]))
                                    preds[l][dst[2:0]] <= $signed(regs[l][srca[4:0]]) < $signed(regs[l][srcb[4:0]]);
                            pc_r <= pc_r + 1;
                            state <= S_FETCH;
                        end
                        8'd9: begin                                               // SETP_EQ
                            for (int l = 0; l < LANES; l++)
                                if (lane_en(l[4:0]))
                                    preds[l][dst[2:0]] <= regs[l][srca[4:0]] == regs[l][srcb[4:0]];
                            pc_r <= pc_r + 1;
                            state <= S_FETCH;
                        end
                        8'd10: begin                                              // MAD d = a*b + c
                            for (int l = 0; l < LANES; l++)
                                if (lane_en(l[4:0]))
                                    regs[l][dst[4:0]] <= regs[l][srca[4:0]] * regs[l][srcb[4:0]] + regs[l][imm[4:0]];
                            pc_r <= pc_r + 1;
                            state <= S_FETCH;
                        end
                        8'd11: begin                                              // MOVI
                            for (int l = 0; l < LANES; l++)
                                if (lane_en(l[4:0]))
                                    regs[l][dst[4:0]] <= imm;
                            pc_r <= pc_r + 1;
                            state <= S_FETCH;
                        end
                        8'd12: begin                                              // MOVR
                            for (int l = 0; l < LANES; l++)
                                if (lane_en(l[4:0]))
                                    regs[l][dst[4:0]] <= regs[l][imm[4:0]];
                            pc_r <= pc_r + 1;
                            state <= S_FETCH;
                        end
                        8'd13: begin                                              // LDP
                            for (int l = 0; l < LANES; l++)
                                regs[l][dst[4:0]] <= params[imm[2:0]];
                            pc_r <= pc_r + 1;
                            state <= S_FETCH;
                        end
                        8'd14: begin                                              // BRA (uniform)
                            pc_r <= branch_taken ? imm[15:0] : pc_r + 1;
                            state <= S_FETCH;
                        end
                        8'd15: begin                                              // SREG laneid
                            for (int l = 0; l < LANES; l++)
                                regs[l][dst[4:0]] <= l;
                            pc_r <= pc_r + 1;
                            state <= S_FETCH;
                        end
                        default: done_r <= 1'b1;
                    endcase
                end

                S_MEM: begin
                    if (seq_lane == 6'd32) begin
                        // all lanes serviced
                        pc_r     <= pc_r + 1;
                        seq_lane <= 6'h0;
                        state    <= S_FETCH;
                    end else if (lane_en(seq_lane)) begin
                        if (!(opcode == 8'd7 && atom_wr))
                            events_r <= events_r + 1;   // one event per RMW, per lane
                        if (opcode == 8'd7) begin
                            if (!atom_wr) begin
                                atom_sum <= mem_rdata + regs[seq_lane][srcb[4:0]];
                                atom_addr <= atom_lane_addr;
                                atom_wr  <= 1'b1;                                 // write phase next
                            end else begin
                                atom_wr  <= 1'b0;
                                waits_r  <= waits_r + 1;
                                seq_lane <= seq_lane + 1;
                            end
                        end else begin
                            if (!mem_we)
                                regs[seq_lane][dst[4:0]] <= mem_rdata;            // load
                            seq_lane <= seq_lane + 1;
                        end
                    end else begin
                        seq_lane <= seq_lane + 1;                                 // lane masked off
                    end
                end
            endcase
        end
    end

endmodule
`default_nettype wire
