#!/usr/bin/env python3
# CTX -> spatial lowering: consume the lifted CTX muops and produce
#   (1) the lowering report: regions (static / token islands / barrier
#       epochs), stream plans for the global-memory groups, and the safe
#       points the runtime may checkpoint at;
#   (2) the instruction ROM for spatial_core (identity schedule: one muop,
#       one ROM use; compaction is explicitly out of scope here).
#
# Pipeline position: PTX --ptx_to_ctx--> CTX --ctx_lower--> ROM --> RTL.
# This mirrors the paper's lowering shape: memory muops in a region are
# grouped into stream plans (base / per-lane address / lane count), an
# ordered atomic becomes a serialized token island, and a barrier muop
# bounds an epoch whose exit is a safe suspension point.

import json
import re
import sys

OP = {"NOP": 0, "EXIT": 1, "LDG": 2, "STG": 3, "ADD": 4, "SHFL": 5,
      "BAR": 6, "ATOM": 7, "SETP_LT": 8, "SETP_EQ": 9, "MAD": 10,
      "MOVI": 11, "MOVR": 12, "LDP": 13, "BRA": 14, "SREG": 15,
      "SELP": 16, "AND": 17}


def rnum(tok):
    tok = tok.strip()
    m = re.fullmatch(r"%r(\d+)", tok)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"%p(\d+)", tok)
    if m:
        return 32 + int(m.group(1))
    if tok in ("0",):
        return 0
    raise ValueError(tok)


def lower(muops, labels):
    rom = []
    report = {"regions": [], "stream_plans": [], "token_islands": [],
              "safe_points": [], "schedule": "identity"}

    # pass 1: muop -> rom index map (bra targets are PTX body indices and
    # each muop records its source ptx index)
    ptx_to_rom = {}
    for m_idx, mu in enumerate(muops):
        if mu.get("ptx") is not None:
            ptx_to_rom.setdefault(mu["ptx"], m_idx)

    region = {"kind": "static", "muops": []}
    streams = {}

    for mu in muops:
        op = mu["op"]
        guard = mu.get("guard", 0)
        neg = mu.get("neg", 0)
        preg = mu.get("preg", 0)
        imm = 0
        dst = a = b = 0

        if op == "split" or op == "merge":
            region["muops"].append(f"{op} ({mu.get('note', '')})")
            continue  # guards are per-muop; the split/merge carries no code

        if op == "sreg.laneid":
            opcode, dst = OP["SREG"], rnum(mu["dst"])
        elif op == "ld.param":
            opcode, dst = OP["LDP"], rnum(mu["dst"])
            imm = mu.get("param_idx", 0)
        elif op == "mov":
            src = mu["imm"]
            if re.fullmatch(r"\d+", src):
                opcode, imm = OP["MOVI"], int(src)
            else:
                opcode, imm = OP["MOVR"], rnum(src)
            dst = rnum(mu["dst"])
        elif op == "ld.global":
            opcode, dst = OP["LDG"], rnum(mu["dst"])
            addr = mu["addr"].strip("[]")
            imm = rnum(addr.split("+")[0])
            # stream plan: per-lane address, base register, full warp width
            key = imm
            streams.setdefault(key, {"kind": "static_affine_stream",
                                     "base_reg": imm, "width": 32,
                                     "events_per_use": 32})
            region["muops"].append(f"ld.global -> stream[{imm}]")
        elif op == "st.global":
            opcode = OP["STG"]
            addr = mu["addr"].strip("[]")
            imm = rnum(addr.split("+")[0])
            a = rnum(mu["val"])
            region["muops"].append(f"st.global -> stream[{imm}]")
        elif op == "add":
            opcode = OP["ADD"]
            dst, a, b = rnum(mu["dst"]), rnum(mu["a"]), rnum(mu["b"])
        elif op == "mad":
            opcode = OP["MAD"]
            dst, a, b = rnum(mu["dst"]), rnum(mu["a"]), rnum(mu["b"])
            imm = rnum(mu["c"])
        elif op == "shfl.down":
            opcode = OP["SHFL"]
            dst, a = rnum(mu["dst"]), rnum(mu["src"])
            imm, b = mu["delta"], mu["clamp"]
            region["muops"].append("shfl.down (permuter use, validity -> %p0)")
        elif op == "bar.sync":
            opcode = OP["BAR"]
            report["safe_points"].append({"rom_index": len(rom),
                                          "kind": "bar.sync"})
            region["muops"].append("bar.sync (epoch boundary, safe point)")
        elif op == "atom.add":
            opcode = OP["ATOM"]
            dst = rnum(mu["dst"])
            addr = mu["addr"].strip("[]")
            base, off = addr.split("+")
            imm, a = rnum(base), rnum(off)
            b = rnum(mu["val"])
            report["token_islands"].append({"rom_index": len(rom),
                                            "kind": "serialized RMW",
                                            "events_per_active_lane": 1})
            region["muops"].append("atom.add -> token island")
        elif op == "selp":
            opcode = OP["SELP"]
            dst, a, b = rnum(mu["dst"]), rnum(mu["a"]), rnum(mu["b"])
            guard, neg, preg = 0, 0, int(mu["pred"][2:])
        elif op == "and":
            opcode = OP["AND"]
            dst, a, b = rnum(mu["dst"]), rnum(mu["a"]), rnum(mu["b"])
        elif op == "setp.lt":
            opcode = OP["SETP_LT"]
            dst, a, b = 32 + int(mu["dst"][2:]), rnum(mu["a"]), rnum(mu["b"])
        elif op == "setp.eq":
            opcode = OP["SETP_EQ"]
            dst, a, b = 32 + int(mu["dst"][2:]), rnum(mu["a"]), rnum(mu["b"])
        elif op == "bra":
            opcode = OP["BRA"]
            # branch target patched in pass 2 (rom_index map)
            dst = a = b = 0
            region["muops"].append(f"bra {mu.get('target')} (uniform control)")
        elif op == "exit":
            opcode = OP["EXIT"]
        else:
            raise ValueError(f"unlowered muop: {op}")

        word_lo = (opcode << 24) | ((dst & 0xFF) << 16) | ((a & 0xFF) << 8) | (b & 0xFF)
        word_hi = ((guard & 1) << 31) | ((neg & 1) << 30) | ((preg & 0x3F) << 24) | (imm & 0xFFFFFF)
        rom.append([word_hi, word_lo])

    # pass 2: patch branch targets (PTX body idx -> rom idx) and restore the
    # guard bit the lifter keeps in the bra muop
    rom_idx_of_muop = {}
    ri = 0
    for m_idx, mu in enumerate(muops):
        if mu["op"] in ("split", "merge"):
            continue
        rom_idx_of_muop[m_idx] = ri
        ri += 1
    for m_idx, mu in enumerate(muops):
        if mu["op"] != "bra":
            continue
        ri2 = rom_idx_of_muop[m_idx]
        tgt_ptx = labels.get(mu.get("target"))
        tgt_rom = rom_idx_of_muop.get(ptx_to_rom.get(tgt_ptx, -1), 0)
        hi, lo = rom[ri2]
        guard = 1 if mu.get("guard", 0) else 0
        hi = ((guard & 1) << 31) | ((mu.get("neg", 0) & 1) << 30) | ((mu.get("preg", 0) & 0x3F) << 24) | (tgt_rom & 0xFFFFFF)
        rom[ri2] = [hi, lo]

    report["regions"].append({"kind": "static", "muops": region["muops"],
                              "note": "identity schedule; token islands and "
                                      "stream plans embedded"})
    report["stream_plans"] = [
        {"base_reg": k, **v} for k, v in streams.items()]
    return rom, report


def main():
    ctx_json, out_hex, out_report, labels_json = (
        sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    data = json.load(open(ctx_json))
    muops = data["muops"] if isinstance(data, dict) else data
    labels = json.load(open(labels_json))["labels"]
    rom, report = lower(muops, labels)
    with open(out_hex, "w") as f:
        for hi, lo in rom:
            f.write(f"{hi & 0xFFFFFFFF:08x} {lo & 0xFFFFFFFF:08x}\n")
    report["n_instr"] = len(rom)
    json.dump(report, open(out_report, "w"), indent=1)
    print(f"lowered: {len(rom)} ROM instructions, "
          f"{len(report['stream_plans'])} stream plans, "
          f"{len(report['token_islands'])} token islands, "
          f"{len(report['safe_points'])} safe points -> {out_hex}")


if __name__ == "__main__":
    main()
