#!/usr/bin/env python3
# DWARF -> CTX state recovery for the NV->TT migration: compile a captured
# kernel PTX with ptxas -g (DWARF emitted), then walk the .debug_info of
# the cubin and recover, per kernel entry, the variables' names, source
# lines, types, and register locations -- exactly the map the CTX state
# annotation carries (paper sec 3.3: "a DWARF-style map links live virtual
# registers, program-counter positions, and active masks to the
# device-neutral snapshot representation").
#
#   dwarf_to_ctx.py <kernel.ptx> [--out ctx_annotation.json]
#
# Output: the recovered map in CTX state-annotation form + the resume
# metadata the runtime consumes at the checkpoint.

import argparse
import json
import os
import subprocess
import sys

from elftools.elf.elffile import ELFFile

PTXAS_CANDIDATES = ["/usr/local/cuda-12.9/bin/ptxas",
                    "/usr/local/cuda-12.8/bin/ptxas", "ptxas"]


def compile_ptx(ptx, workdir):
    ptxas = next(p for p in PTXAS_CANDIDATES
                 if subprocess.run(["bash", "-c", f"command -v {p}"],
                                   capture_output=True).returncode == 0
                 or os.path.exists(p))
    cubin = os.path.join(workdir, "kernel.cubin")
    target = "sm_120"
    for line in open(ptx):
        if line.startswith(".target"):
            raw = line.split("sm_")[1].split()[0]
            target = f"sm_{raw}"      # keep the suffix (sm_120a etc.)
            break
    r = subprocess.run([ptxas, "-g", "-arch", target, "-o", cubin, ptx],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ptxas failed: {r.stderr[:400]}")
    return cubin, target


def mask_relocations(path):
    """pyelftools cannot decode cubin's exotic RELA types; flip every
    SHT_RELA section header to SHT_NOBITS so the DWARF consumer ignores
    them. The names/lines/types we recover are relocation-independent."""
    import struct
    data = bytearray(open(path, "rb").read())
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize = struct.unpack_from("<H", data, 0x3A)[0]
    e_shnum = struct.unpack_from("<H", data, 0x3C)[0]
    masked = 0
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh_type = struct.unpack_from("<I", data, off + 4)[0]
        if sh_type == 4:  # SHT_RELA
            struct.pack_into("<I", data, off + 4, 8)  # SHT_NOBITS
            masked += 1
    open(path, "wb").write(data)
    return masked


def walk_dwarf(cubin):
    """Recover the variable/position map from .debug_info + .debug_line."""
    out = {"entries": [], "line_program": None}
    with open(cubin, "rb") as f:
        elf = ELFFile(f)
        if not elf.has_dwarf_info():
            raise RuntimeError("cubin carries no DWARF")
        dw = elf.get_dwarf_info()
        cus = list(dw.iter_CUs())
        for cu in cus:
            top = cu.get_top_DIE()
            entry = {"name": top.attributes["DW_AT_name"].value.decode()
                     if "DW_AT_name" in top.attributes else "?",
                     "comp_dir": top.attributes.get("DW_AT_comp_dir",
                        None) and top.attributes["DW_AT_comp_dir"].value.decode(),
                     "producer": top.attributes.get("DW_AT_producer",
                        None) and top.attributes["DW_AT_producer"].value.decode(),
                     "language": str(top.attributes.get("DW_AT_language", "").value or ""),
                     "variables": []}
            subprograms = []
            for die in cu.iter_DIEs():
                if die.tag == "DW_TAG_subprogram":
                    sp = {"name": die.attributes.get("DW_AT_name",
                          None) and die.attributes["DW_AT_name"].value.decode()}
                    lo = die.attributes.get("DW_AT_low_pc")
                    hi = die.attributes.get("DW_AT_high_pc")
                    if lo is not None:
                        sp["low_pc"] = lo.value
                        sp["high_pc"] = hi.value if hi else None
                    subprograms.append(sp)
                    continue
                if die.tag != "DW_TAG_variable":
                    continue
                name = die.attributes.get("DW_AT_name")
                decl = die.attributes.get("DW_AT_decl_line")
                typ = die.attributes.get("DW_AT_type")
                loc = die.attributes.get("DW_AT_location")
                decl_file = die.attributes.get("DW_AT_decl_file")
                v = {"name": name.value.decode() if name else "?",
                     "decl_line": decl.value if decl else None,
                     "decl_file": decl_file.value if decl_file else None,
                     "type_ref": typ.value if typ else None,
                     "location": None}
                if loc:
                    try:
                        blk = loc.value
                        if isinstance(blk, list):
                            blk = bytes(blk)
                        v["location"] = blk.hex()
                    except Exception:
                        v["location"] = "exprloc"
                # resolve type name via the CU's type DIE
                if typ is not None:
                    try:
                        t_die = cu.get_DIE_from_refaddr(typ.value + cu.cu_offset)
                        v["type"] = t_die.attributes.get("DW_AT_name",
                            None) and t_die.attributes["DW_AT_name"].value.decode()
                    except Exception:
                        v["type"] = None
                entry["variables"].append(v)
            entry["subprograms"] = subprograms
            out["entries"].append(entry)
        ptx_sec = elf.get_section_by_name(".nv_debug_ptx_txt")
        if ptx_sec is not None:
            txt = ptx_sec.data()
            out["embedded_ptx_bytes"] = len(txt)
            out["embedded_ptx_head"] = txt[:160].decode("utf-8", "replace")
        lps = [dw.line_program_for_CU(cus[0])] if cus else []
        if lps and lps[0] is not None:
            lp = lps[0]
            header = lp.header
            names = [f.name.decode() for f in header.file_entry]
            out["line_program"] = {
                "version": header.version,
                "file": names[0] if names else "?",
            }
            rows = []
            for lentry in lp.get_entries():
                de = lentry.state
                if de is None:
                    continue
                rows.append({"addr": de.address, "line": de.line})
                if len(rows) >= 16:
                    break
            out["line_program"]["rows_first16"] = rows
    return out


def to_ctx_annotation(recovered, ptx_name):
    """Render the DWARF map as the CTX state annotation the runtime ships
    with a checkpoint."""
    e = recovered["entries"][0] if recovered["entries"] else {}
    return {
        "format": "device-neutral/ctx-state-v1",
        "kind": "dwarf-recovered-value-map",
        "kernel": e.get("name", "?"),
        "producer": e.get("producer", ""),
        "compilation_unit": {"file": e.get("name"), "dir": e.get("comp_dir")},
        "line_program": recovered.get("line_program"),
        "value_map": [
            {"vreg_or_var": v["name"], "ptx_line": v["decl_line"],
             "type": v.get("type"), "location": v["location"]}
            for v in e.get("variables", [])],
        "kernel_pc_ranges": e.get("subprograms", []),
        "safe_point": "kernel entry (resume mapping: pc 0)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ptx")
    ap.add_argument("--out", default="results/ctx_annotation.json")
    ap.add_argument("--workdir", default="/tmp/dwarf_ctx")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cubin, target = compile_ptx(args.ptx, args.workdir)
    n_masked = mask_relocations(cubin)
    recovered = walk_dwarf(cubin)
    ann = to_ctx_annotation(recovered, os.path.basename(args.ptx))
    ann["target"] = target
    json.dump(ann, open(args.out, "w"), indent=1)
    nvars = len(ann["value_map"])
    print(f"{os.path.basename(args.ptx)}: target {target} "
          f"({n_masked} reloc sections masked), "
          f"DWARF cu '{ann['kernel']}', {nvars} variables recovered, "
          f"line program v{recovered['line_program']['version']} "
          f"({recovered['line_program']['file']}) -> {args.out}")


if __name__ == "__main__":
    main()
