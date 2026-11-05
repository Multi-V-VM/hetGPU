#!/usr/bin/env python3
# Transliterate the supported PTX fragment into spatial-core ROM words.
#
# Deterministic front end of the end-to-end experiment: parse the kernel
# PTX, resolve labels, emit two 32-bit words per instruction.
#
#   word_lo = { op[7:0], dst[7:0], a[7:0], b[7:0] }
#   word_hi = { guard[1], neg[1], preg[6], imm[23:0] }
#
#   guard=0 : unconditional      guard=1 : execute under predicate preg
#   neg=1   : predicate negated (@!p)
#
# Register encoding: %rN -> N (0..23), predicate dst %pN -> 32+N.
# Opcodes match spatial_core.sv:
#   0 NOP  1 EXIT  2 LDG  3 STG  4 ADD  5 SHFL  6 BAR  7 ATOM
#   8 SETP_LT  9 SETP_EQ  10 MAD  11 MOVI  12 MOVR  13 LDP  14 BRA  15 SREG

import json
import re
import sys

OP = {"NOP": 0, "EXIT": 1, "LDG": 2, "STG": 3, "ADD": 4, "SHFL": 5,
      "BAR": 6, "ATOM": 7, "SETP_LT": 8, "SETP_EQ": 9, "MAD": 10,
      "MOVI": 11, "MOVR": 12, "LDP": 13, "BRA": 14, "SREG": 15}


def parse_lines(path):
    """Return instruction lines with labels resolved into a table."""
    labels = {}
    params = {}
    body = []
    for raw in open(path):
        text = raw.split("//")[0]
        for stmt in text.split(";"):
            s = stmt.strip()
            if not s:
                continue
            if s.startswith(".version") or s.startswith(".target") or s.startswith(".reg"):
                continue
            if s.startswith(".visible") or s == "{" or s == "}":
                for m in re.finditer(r"\.param\s*\.u(?:32|64)\s+(\w+)", s):
                    params[m.group(1)] = len(params)
                continue
            if s.startswith("."):
                continue
            if s.endswith(":") and " " not in s:
                labels[s[:-1]] = len(body)
                continue
            body.append(s)
    return body, labels, params


def rnum(tok):
    m = re.fullmatch(r"%r(\d+)", tok.strip())
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"%p(\d+)", tok.strip())
    if m:
        return 32 + int(m.group(1))
    raise ValueError(f"not a register: {tok}")


def pnum(tok):
    return int(re.fullmatch(r"%p(\d+)", tok.strip()).group(1))


def translit(body, labels, params):
    rom = []
    branch_patches = []
    for line in body:
        guard, neg, preg = 0, 0, 0
        m = re.match(r"@(!?)%(p\d+)\s+(.*)", line)
        if m:
            guard, neg, preg = 1, 1 if m.group(1) else 0, int(m.group(2)[1:])
            line = m.group(3)
        parts = [t for t in re.split(r"[,\s]+", line) if t]
        head = parts[0].split(".")[0]
        opfull = parts[0]
        op, dst, a, b, imm = 0, 0, 0, 0, 0

        if head == "mov":
            dst = rnum(parts[1])
            src = parts[2]
            if "%laneid" in src:
                op = OP["SREG"]
            elif re.fullmatch(r"\d+", src):
                op, imm = OP["MOVI"], int(src)
            else:
                op, imm = OP["MOVR"], rnum(src)
        elif head == "ld":
            if ".param" in opfull:
                op = OP["LDP"]
                dst = rnum(parts[1])
                pname = parts[2].split(".")[-1]
                imm = params[pname]
            elif ".global" in opfull:
                op = OP["LDG"]
                dst = rnum(parts[1])
                addr = line[line.index("[") + 1:line.index("]")]
                imm = rnum(addr.split("+")[0])
        elif head == "st":
            op = OP["STG"]
            addr = line[line.index("[") + 1:line.index("]")]
            imm = rnum(addr.split("+")[0])
            a = rnum(parts[-1])
        elif head == "add":
            op = OP["ADD"]
            dst, a, b = rnum(parts[1]), rnum(parts[2]), rnum(parts[3])
        elif head == "mad":
            op = OP["MAD"]
            dst, a, b = rnum(parts[1]), rnum(parts[2]), rnum(parts[3])
            imm = rnum(parts[4])
        elif head == "shfl":
            op = OP["SHFL"]
            dst, a = rnum(parts[1]), rnum(parts[2])
            imm = int(parts[3])          # delta
            b = int(parts[4], 0)         # clamp (segment not modeled: 31 = full)
        elif head == "bar":
            op = OP["BAR"]
        elif head == "atom":
            op = OP["ATOM"]
            dst = rnum(parts[1])
            addr = parts[2].strip("[]")
            base, off = addr.split("+")
            imm, a = rnum(base), rnum(off)
            b = rnum(parts[3])           # value operand
        elif head == "setp":
            op = OP["SETP_LT"] if ".lt" in opfull else OP["SETP_EQ"]
            dst = 32 + pnum(parts[1])
            a, b = rnum(parts[2]), rnum(parts[3])
        elif head == "bra":
            op = OP["BRA"]
            branch_patches.append((len(rom), parts[1]))
        elif head == "ret":
            op = OP["EXIT"]
        else:
            raise ValueError(f"unsupported instruction: {line}")

        word_lo = (op << 24) | ((dst & 0xFF) << 16) | ((a & 0xFF) << 8) | (b & 0xFF)
        rom.append([word_hi_of(guard, neg, preg, imm), word_lo])

    for idx, lbl in branch_patches:
        rom[idx][0] = word_hi_of(rom[idx][0] >> 31, (rom[idx][0] >> 30) & 1,
                                 (rom[idx][0] >> 24) & 0x3F, labels[lbl])
    return rom


def word_hi_of(guard, neg, preg, imm):
    return ((guard & 1) << 31) | ((neg & 1) << 30) | ((preg & 0x3F) << 24) | (imm & 0xFFFFFF)


def main():
    ptx, out_hex, out_meta = sys.argv[1], sys.argv[2], sys.argv[3]
    body, labels, params = parse_lines(ptx)
    rom = translit(body, labels, params)
    with open(out_hex, "w") as f:
        for hi, lo in rom:
            f.write(f"{hi & 0xFFFFFFFF:08x} {lo & 0xFFFFFFFF:08x}\n")
    meta = {"n_instr": len(rom), "labels": labels, "params": params}
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"rom: {len(rom)} instructions -> {out_hex}")


if __name__ == "__main__":
    main()
