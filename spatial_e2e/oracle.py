#!/usr/bin/env python3
# Golden oracle for the spatial end-to-end experiment: executes the same
# transliterated ROM the RTL runs, bit-exact (int32 wraparound), and checks
# the Verilator run's final memory against it.
#
#   oracle.py expected --rom rom_mm.hex [--tiles 8 --tailw 20 --seed N]
#   oracle.py check    --rom rom_mm.hex --mem <run.json.mem> [...]
#
# Cycle model mirrors the RTL: one EXEC cycle per instruction plus one cycle
# per serviced lane for memory ops (two per atomic lane: RMW read+write).

import argparse
import json
import struct
import sys

M32 = 0xFFFFFFFF
MEM_WORDS = 1 << 16


def s32(v):
    v &= M32
    return v - (1 << 32) if v & 0x80000000 else v


def u32(v):
    return v & M32


def load_rom(path):
    return [int(w, 16) for w in open(path).read().split()]


def make_inputs(tiles, tailw, seed):
    s = seed
    def lcg():
        nonlocal s
        s = u32((s * 6364136223846793005 + 1442695040888963407) & 0x1FFFFFFFFFFFFFFFF) >> 0
        # 64-bit LCG, take high 32 bits like the C++ side
        return (s >> 33) & M32
    # match tb_main.cpp: s = s*A + C (mod 2^64), out = (s >> 33)
    st = seed
    def lcgc():
        nonlocal st
        st = (st * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (st >> 33) & M32
    A = [(lcgc() % 2000) - 1000 for _ in range(tiles * 32)]
    B = [(lcgc() % 2000) - 1000 for _ in range(32)]
    mem = [0] * MEM_WORDS
    for i, v in enumerate(A):
        mem[i] = u32(v)
    for i, v in enumerate(B):
        mem[4096 + i] = u32(v)
    return mem


def run(rom, tiles, tailw, seed, base_a=0, base_b=4096, base_c=8192,
        checkpoint_pc=None):
    mem = make_inputs(tiles, tailw, seed)
    regs = [[0] * 32 for _ in range(32)]
    preds = [[False] * 8 for _ in range(32)]
    pc, cycles, events, waits = 0, 0, 0, 0
    stopped = False

    def en(l, guard, neg, preg):
        if guard == 0:
            return True
        p = preds[l][preg]
        return (not p) if neg else p

    n_instr = len(rom)
    while True:
        if checkpoint_pc is not None and pc == checkpoint_pc and cycles > 0:
            stopped = True
            break
        hi, lo = rom[2 * pc], rom[2 * pc + 1]
        op = (lo >> 24) & 0xFF
        dst = (lo >> 16) & 0xFF
        srca = (lo >> 8) & 0xFF
        srcb = lo & 0xFF
        guard = (hi >> 31) & 1
        neg = (hi >> 30) & 1
        preg = (hi >> 24) & 0x3F
        imm = hi & 0xFFFFFF
        cycles += 2  # FETCH + EXEC

        if op == 1:  # EXIT
            break
        elif op == 0 or op == 6:  # NOP / BAR
            pc += 1
        elif op == 14:  # BRA uniform (all lanes agree by construction in this fragment)
            taken = (not preds[0][preg]) if neg else preds[0][preg]
            if guard == 0:
                taken = True
            pc = imm if taken else pc + 1
        elif op == 4:  # ADD
            for l in range(32):
                if en(l, guard, neg, preg):
                    regs[l][dst] = u32(regs[l][srca] + regs[l][srcb])
            pc += 1
        elif op == 10:  # MAD
            for l in range(32):
                if en(l, guard, neg, preg):
                    regs[l][dst] = u32(regs[l][srca] * regs[l][srcb] + regs[l][imm])
            pc += 1
        elif op == 11:
            for l in range(32):
                if en(l, guard, neg, preg):
                    regs[l][dst] = imm
            pc += 1
        elif op == 12:
            for l in range(32):
                if en(l, guard, neg, preg):
                    regs[l][dst] = regs[l][imm]
            pc += 1
        elif op == 13:
            params = [base_a, base_b, base_c, tiles, tailw, 0, 0, 0]
            for l in range(32):
                regs[l][dst] = params[imm]
            pc += 1
        elif op == 15:
            for l in range(32):
                regs[l][dst] = l
            pc += 1
        elif op == 8:  # SETP_LT (signed)
            for l in range(32):
                if en(l, guard, neg, preg):
                    preds[l][dst - 32] = s32(regs[l][srca]) < s32(regs[l][srcb])
            pc += 1
        elif op == 9:  # SETP_EQ
            for l in range(32):
                if en(l, guard, neg, preg):
                    preds[l][dst - 32] = regs[l][srca] == regs[l][srcb]
            pc += 1
        elif op == 5:  # SHFL_DOWN
            for l in range(32):
                src = l + imm
                if src <= srcb:
                    regs[l][dst] = regs[src][srca]
                    preds[l][0] = True
                else:
                    regs[l][dst] = regs[l][srca]
                    preds[l][0] = False
            pc += 1
        elif op == 2:  # LDG: every visited lane costs a cycle, enabled or not
            for l in range(32):
                if en(l, guard, neg, preg):
                    regs[l][dst] = mem[regs[l][imm]]
                    events += 1
            cycles += 33  # 32 lane slots + all-serviced wrap-up
            pc += 1
        elif op == 3:  # STG
            for l in range(32):
                if en(l, guard, neg, preg):
                    mem[regs[l][imm]] = regs[l][srca]
                    events += 1
            cycles += 33
            pc += 1
        elif op == 7:  # ATOM_ADD island: enabled lane = read+write (2 cycles,
                # 1 event, 1 token wait); a masked lane costs one idle slot
            for l in range(32):
                if en(l, guard, neg, preg):
                    addr = u32(regs[l][imm] + regs[l][srca])
                    old = mem[addr]
                    mem[addr] = u32(old + regs[l][srcb])
                    regs[l][dst] = old
                    events += 1
                    waits += 1
                    cycles += 2
                else:
                    cycles += 1
            cycles += 1  # wrap-up
            pc += 1
        else:
            raise ValueError(f"op {op} at pc {pc}")

    checksums = [u32(sum(regs[l])) for l in range(32)]
    return {
        "stopped_at_checkpoint": stopped,
        "checkpoint_pc": checkpoint_pc,
        "pc": pc,
        "cycles": cycles,
        "mem_events": events,
        "token_waits": waits,
        "C": [s32(mem[base_c + i]) for i in range(tiles)],
        "sentinel": s32(mem[base_c + tiles]),
        "checksums": checksums,
        "state": (regs, preds, pc, mem, cycles, events, waits) if stopped else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["expected", "check"])
    ap.add_argument("--rom", default="rom_mm.hex")
    ap.add_argument("--tiles", type=int, default=8)
    ap.add_argument("--tailw", type=int, default=20)
    ap.add_argument("--seed", type=lambda x: int(x, 0), default=20261105)
    ap.add_argument("--mem")
    ap.add_argument("--run-json")
    ap.add_argument("--at-pc", type=int, default=None)
    args = ap.parse_args()

    rom = load_rom(args.rom)
    r = run(rom, args.tiles, args.tailw, args.seed, checkpoint_pc=args.at_pc)

    if args.cmd == "expected":
        print(json.dumps({k: v for k, v in r.items() if k != "state"}, indent=1))
        return

    # check mode: compare a Verilator run's final memory against the oracle
    mem = struct.unpack(f"<{MEM_WORDS}i", open(args.mem, "rb").read())
    ok = True
    for i in range(args.tiles):
        got, exp = s32(u32(mem[8192 + i])), r["C"][i]
        if got != exp:
            print(f"C[{i}]: got {got} expected {exp}")
            ok = False
    got_s, exp_s = s32(u32(mem[8192 + args.tiles])), r["sentinel"]
    if got_s != exp_s:
        print(f"sentinel: got {got_s} expected {exp_s}")
        ok = False
    print("ORACLE_MATCH" if ok else "ORACLE_MISMATCH")
    print(f"expected: cycles={r['cycles']} events={r['mem_events']} "
          f"waits={r['token_waits']}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
