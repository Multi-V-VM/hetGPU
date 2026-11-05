#!/usr/bin/env python3
# PTX -> CTX: lift the supported PTX fragment into CTX, the paper's
# serialized muop form (sec 3.3 of the paper). CTX makes what the PTX
# leaves implicit explicit:
#   - every value is a lane-packed vector with an active-lane mask
#   - every effect (load/store/atom) carries its guard, state space, and
#     order/scope labels
#   - every barrier is a labeled safe suspension point
#   - short guarded diamonds are if-converted: the guard region is a
#     mask split (SAVE/SETM) and the join is a mask merge (RESTM)
#
# Output: a CTX module (text, the notation of the paper's Figure 2) plus a
# JSON sidecar describing safe points and the muop inventory -- the input
# of ctx_lower.py, the spatial lowering.

import json
import re
import sys

# ---------------------------------------------------------------- lifting

def parse_ptx(path):
    labels, params, body = {}, {}, []
    for raw in open(path):
        text = raw.split("//")[0]
        for stmt in text.split(";"):
            s = stmt.strip()
            if not s:
                continue
            if s.startswith((".version", ".target", ".reg")):
                continue
            if s.startswith(".visible") or s in ("{", "}"):
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


def lift(body, labels, params):
    lift_stmt.params = params
    """Emit CTX muop records. Short guarded diamonds (`@p bra L1; fall;
    L1:`) are if-converted into SETM/RESTM around the fall path."""
    ctx = []
    pc = 0
    safepoints = []
    i = 0
    n = len(body)
    while i < n:
        line = body[i]
        # --- guarded branches: if-convert short diamonds, else control muop
        m = re.match(r"@(!?)%(p\d+)\s+bra\s+(\w+)$", line)
        if m:
            neg, preg, target = m.group(1) == "!", int(m.group(2)[1:]), m.group(3)
            nxt = i + 1
            tgt = labels.get(target, -1)
            # forward target: if-convert only SHORT guarded diamonds; far
            # branches (loop exits) stay explicit control muops.
            # Shape: @p bra A; <fall...>; [bra J;] A: <taken...>; J:
            if 0 <= tgt and nxt < tgt <= nxt + 8:
                # find the fall arm's unconditional terminator bra J (if any)
                jbra, join = None, tgt
                for j in range(nxt, tgt):
                    m2 = re.match(r"bra\s+(\w+)$", body[j])
                    if m2:
                        jbra, join = j, labels.get(m2.group(1), -1)
                        break
                if jbra is not None and join > tgt:
                    # two-arm diamond: fall = [nxt, jbra) under !p,
                    # taken = [tgt, join) under p
                    ctx.append({"op": "split", "preg": preg, "neg": neg,
                                "src": f"ptx:{i}", "note": "if-convert two-arm diamond"})
                    pcn = pc + 1
                    for j in range(nxt, jbra):
                        mu = lift_stmt(body[j], pcn)
                        mu["arm"], mu["guard"], mu["neg"], mu["preg"] = "fall", 1, not neg, preg
                        ctx.append(mu); pcn += 1
                    for j in range(tgt, join):
                        mu = lift_stmt(body[j], pcn)
                        mu["arm"], mu["guard"], mu["neg"], mu["preg"] = "taken", 1, neg, preg
                        ctx.append(mu); pcn += 1
                    ctx.append({"op": "merge", "src": f"ptx:{join}", "note": "region join"})
                    pc += 2 + (jbra - nxt) + (join - tgt)
                    i = join
                    continue
                # plain skip guard: skipped region executes under !p
                join = tgt
                ctx.append({"op": "split", "preg": preg, "neg": neg,
                            "src": f"ptx:{i}", "note": "if-convert region"})
                pcn = pc + 1
                for j in range(nxt, join):
                    if re.match(r"bra\s+\w+$", body[j]):
                        continue  # region terminator = the merge
                    mu = lift_stmt(body[j], pcn)
                    mu["arm"], mu["guard"], mu["neg"], mu["preg"] = "fall", 1, not neg, preg
                    ctx.append(mu); pcn += 1
                ctx.append({"op": "merge", "src": f"ptx:{join}", "note": "region join"})
                pc += 2 + (join - nxt)
                i = join
                continue
            # far / backward target: uniform control muop (guarded)
            ctx.append({"op": "bra", "guard": 1, "preg": preg, "neg": neg,
                        "target": target, "ptx": i, "src": f"ptx:{i}"})
            pc += 1
            i += 1
            continue
        ctx.append(lift_stmt(line, pc, i))
        if line.split()[0].split(".")[0] == "bar":
            safepoints.append({"pc": pc, "kind": "bar.sync",
                               "label": f"SP{len(safepoints)}"})
        pc += 1
        i += 1
    return ctx, safepoints


def lift_stmt(line, pc, ptx_idx=None, params=None):
    guard, neg, preg = 0, 0, 0
    m = re.match(r"@(!?)%(p\d+)\s+(.*)", line)
    if m:
        guard, neg, preg = 1, m.group(1) == "!", int(m.group(2)[1:])
        line = m.group(3)
    parts = [t for t in re.split(r"[,\s]+", line) if t]
    head = parts[0].split(".")[0]
    opfull = parts[0]
    mu = {"pc": pc, "guard": guard, "neg": neg, "preg": preg,
          "ptx": ptx_idx, "src": line.strip()}
    if head == "mov":
        mu["op"] = "sreg.laneid" if "%laneid" in line else "mov"
        mu["dst"] = parts[1]
        mu["imm"] = parts[2] if "%laneid" not in line else "laneid"
    elif head == "ld":
        if ".param" in opfull:
            mu["op"] = "ld.param"
            mu["dst"], mu["param"] = parts[1], parts[2].split(".")[-1]
            mu["param_idx"] = (lift_stmt.params or {}).get(mu["param"], 0)
        else:
            mu["op"] = "ld.global"
            mu["space"] = "global"
            mu["order"], mu["scope"] = "weak", "none"
            mu["dst"] = parts[1]
            mu["addr"] = parts[2]
    elif head == "st":
        mu["op"] = "st.global"
        mu["space"] = "global"
        mu["order"], mu["scope"] = "weak", "none"
        mu["addr"] = parts[1]
        mu["val"] = parts[-1]
    elif head == "add":
        mu["op"] = "add"
        mu["dst"], mu["a"], mu["b"] = parts[1], parts[2], parts[3]
    elif head == "mad":
        mu["op"] = "mad"
        mu["dst"], mu["a"], mu["b"], mu["c"] = parts[1], parts[2], parts[3], parts[4]
    elif head == "shfl":
        mu["op"] = "shfl.down"
        mu["dst"], mu["src"] = parts[1], parts[2]
        mu["delta"], mu["clamp"] = int(parts[3]), int(parts[4], 0)
        mu["membermask"] = parts[5] if len(parts) > 5 else "0xffffffff"
    elif head == "bar":
        mu["op"] = "bar.sync"
    elif head == "atom":
        mu["op"] = "atom.add"
        mu["space"] = "global"
        mu["order"], mu["scope"] = "relaxed", "gpu"
        mu["dst"] = parts[1]
        mu["addr"], mu["val"] = parts[2], parts[3]
    elif head == "setp":
        mu["op"] = "setp.lt" if ".lt" in opfull else "setp.eq"
        mu["dst"], mu["a"], mu["b"] = parts[1], parts[2], parts[3]
    elif head == "bra":
        mu["op"] = "bra"
        mu["target"] = parts[1]
    elif head == "ret":
        mu["op"] = "exit"
    else:
        raise ValueError(f"unsupported: {line}")
    return mu


def render(muops, labels, safepoints):
    """CTX text in the notation of the paper's Figure 2."""
    out = ["; CTX: lane-packed mask-SSA muops lifted from PTX",
           "; W = 32 lanes; guards are active-lane masks; memory muops carry",
           "; state space, order, and scope; bar.sync muops are safe points."]
    sp_at = {s["pc"]: s["label"] for s in safepoints}
    for mu in muops:
        g = ""
        if mu.get("guard"):
            g = f"@{'!' if mu['neg'] else ''}%p{mu['preg']} "
            if mu.get("arm"):
                g += f"[{mu['arm']}] "
        mpc = mu.get("pc")
        tag = f"sp:{sp_at.get(mpc, '')}" if mpc is not None and mpc in sp_at else ""
        src = mu.get("src", "")
        if "pc" not in mu:
            mu["pc"] = -1
        if mu["op"] == "split":
            out.append(f"  {g}split.mask  %p{mu['preg']}{' (neg)' if mu['neg'] else ''}   ; {src}")
        elif mu["op"] == "merge":
            out.append(f"  merge.mask                ; {src}")
        elif mu["op"] == "bra":
            g2 = f"@{'!' if mu.get('neg') else ''}%p{mu['preg']} " if mu.get("guard") else ""
            out.append(f"  {g2}bra        {mu.get('target', '')}   ; {src}")
        elif mu["op"] == "ld.global":
            out.append(f"  {g}{mu['op']:<10s} {mu['dst']}, [{mu['addr']}]  space=global order={mu['order']}   ; {src}")
        elif mu["op"] == "st.global":
            out.append(f"  {g}{mu['op']:<10s} [{mu['addr']}], {mu['val']}  space=global order={mu['order']}   ; {src}")
        elif mu["op"] == "atom.add":
            out.append(f"  {g}{mu['op']:<10s} {mu['dst']}, {mu['addr']}, {mu['val']}  space=global order={mu['order']} scope={mu['scope']}   ; {src}")
        elif mu["op"] == "shfl.down":
            out.append(f"  {g}{mu['op']:<10s} {mu['dst']}, {mu['src']}, {mu['delta']}, {mu['clamp']}, {mu['membermask']}   ; {src}")
        elif mu["op"] == "bar.sync":
            out.append(f"  bar.sync   0     ; SAFE POINT {tag}")
        else:
            fields = ", ".join(f"{k}={v}" for k, v in mu.items()
                               if k not in ("op", "pc", "guard", "neg", "preg", "src"))
            out.append(f"  {g}{mu['op']:<10s} {fields}   ; {src}")
    return "\n".join(out)


def main():
    ptx, out_ctx, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    body, labels, params = parse_ptx(ptx)
    muops, safepoints = lift(body, labels, params)
    text = render(muops, labels, safepoints)
    open(out_ctx, "w").write(text + "\n")
    json.dump({"muops": muops, "labels": labels, "params": params,
               "safepoints": safepoints, "n_muops": len(muops)},
              open(out_json, "w"), indent=1)
    print(f"CTX: {len(muops)} muops, {len(safepoints)} safe points -> {out_ctx}")


if __name__ == "__main__":
    main()
