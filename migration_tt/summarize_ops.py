#!/usr/bin/env python
# Summarize the TT->TT migration matrix: verify bit-identity across
# baseline / instrumented / migrated runs, compose migrated totals across
# the process boundary, and compute slowdowns.
import json, glob, os
import torch

R = "results/ops"

def load(kind, kern, rep):
    p = f"{R}/{kind}_{kern}_{rep}.json"
    return json.load(open(p)) if os.path.exists(p) else None

def final_file(kind, kern, rep):
    # the harness wrote finals into results/ (mode name spelled out)
    mode = {"base": "baseline", "inst": "instrumented", "tgt": "target"}[kind]
    return f"results/final_{mode}_{kern}_{rep}.pt"

rows = []
bitident = True
for kern in ("matmul", "vecadd", "pi"):
    for rep in (0, 1, 2):
        base, inst, src, tgt = (load(k, kern, rep) for k in
                                ("base", "inst", "src", "tgt"))
        if not all((base, inst, src, tgt)):
            print(f"MISSING {kern} rep{rep}"); bitident = False; continue

        # --- verification: migrated run must reproduce the uninterrupted run
        f_ref  = torch.load(final_file("base", kern, rep), weights_only=False)
        f_inst = torch.load(final_file("inst", kern, rep), weights_only=False)
        f_tgt  = torch.load(final_file("tgt", kern, rep), weights_only=False)
        eq_i = all(torch.equal(f_ref[k2], f_inst[k2]) for k2 in f_ref)
        eq_m = all(torch.equal(f_ref[k2], f_tgt[k2]) for k2 in f_ref)
        # cross-rep determinism of the baseline itself
        if rep > 0:
            f_prev = torch.load(final_file("base", kern, rep - 1), weights_only=False)
            eq_i = eq_i and all(torch.equal(f_prev[k2], f_ref[k2]) for k2 in f_ref)
        bitident &= (eq_i and eq_m)

        migrated_total = src["total_s"] + tgt["total_s"]
        downtime = (tgt["first_resume_done_ns"] - src["capture_start_ns"]) / 1e9
        rows.append({
            "kernel": kern, "rep": rep,
            "baseline_s": base["total_s"],
            "instrumented_s": inst["total_s"],
            "migrated_s": round(migrated_total, 4),
            "downtime_s": round(downtime, 4),
            "src_total_s": src["total_s"], "tgt_total_s": tgt["total_s"],
            "state_bytes": tgt["state_bytes"],
            "capture_per_batch_s": inst["timings"].get("capture", 0) / base.get("n_batches", 1),
            "tgt_create_device_s": tgt["create_device_s"],
            "tgt_first_batch_s": tgt.get("first_batch_s"),
            "slowdown_vs_base_pct": round((migrated_total / base["total_s"] - 1) * 100, 2),
            "instr_overhead_pct": round((inst["total_s"] / base["total_s"] - 1) * 100, 2),
            "bitident_inst": eq_i, "bitident_mig": eq_m,
            "check": {k: (base.get("checksum"), inst.get("checksum"),
                          tgt.get("checksum")) for k in ("checksum",) if base.get("checksum")},
            "pi": [base.get("pi_est"), inst.get("pi_est"), tgt.get("pi_est")],
            "hits": [base.get("hits"), inst.get("hits"), tgt.get("hits")],
        })

def mean(xs): return sum(xs) / len(xs)

out = {"rows": rows, "bitident_all": bitident}
print(f"{'kernel':8s} {'base':>8s} {'instr':>8s} {'migr':>8s} {'downtime':>9s} "
      f"{'slow%':>8s} {'instr%':>8s} {'ident':>6s}")
for kern in ("matmul", "vecadd", "pi"):
    rs = [r for r in rows if r["kernel"] == kern]
    if not rs: continue
    print(f"{kern:8s} {mean([r['baseline_s'] for r in rs]):8.2f} "
          f"{mean([r['instrumented_s'] for r in rs]):8.2f} "
          f"{mean([r['migrated_s'] for r in rs]):8.2f} "
          f"{mean([r['downtime_s'] for r in rs]):9.3f} "
          f"{mean([r['slowdown_vs_base_pct'] for r in rs]):8.2f} "
          f"{mean([r['instr_overhead_pct'] for r in rs]):8.2f} "
          f"{str(all(r['bitident_mig'] and r['bitident_inst'] for r in rs)):>6s}")

json.dump(out, open(f"{R}/summary.json", "w"), indent=1)
print("\nbit_identity_all:", bitident)
