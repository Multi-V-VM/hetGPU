#!/usr/bin/env python
# TT->spatial migration bridge, source stage: run K batches of the int32
# vector-add accumulation on a simulated Tenstorrent Blackhole (ttnn), then
# serialize the device-neutral state -- the CTX-level checkpoint: live
# values plus the annotations naming where the state was taken.
#
#   tt_to_spatial.py full    --batches N            -> results C_full
#   tt_to_spatial.py source  --k K --batches N --out state.npz
#
# The spatial stage (run_spatial.sh) restores C/A into the RTL core's
# memory and executes the remaining N-K batches; bit-identity against the
# full-TT run is the feasibility proof.

import argparse
import json
import os

import numpy as np
import torch

SEED = 0xC0FFEE
ELEMS = 4096


def run_tt(batches, c_init=None, a_fixed=None):
    import ttnn
    dev = ttnn.CreateDevice(device_id=0)
    g = torch.Generator().manual_seed(SEED)
    c = (torch.randint(0, 1000, (1, ELEMS), generator=g, dtype=torch.int32)
         if c_init is None else c_init.clone())
    a = (torch.randint(1, 1000, (1, ELEMS), generator=g, dtype=torch.int32)
         if a_fixed is None else a_fixed.clone())
    ta = ttnn.from_torch(a, dtype=ttnn.int32, device=dev,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tc = ttnn.from_torch(c, dtype=ttnn.int32, device=dev,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for _ in range(batches):
        tc = ttnn.add(tc, ta, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.to_torch(tc).to(torch.int32)
    ttnn.CloseDevice(dev)
    return c.numpy(), a.numpy(), out.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["full", "source", "verify"])
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="results/tt_state.npz")
    ap.add_argument("--spatial-mem")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    if args.mode == "full":
        c0, a, c_full = run_tt(args.batches)
        np.savez("results/tt_full.npz", c0=c0, a=a, c_full=c_full)
        print(f"full TT run: {args.batches} batches, "
              f"C[0:4]={c_full[0, :4].tolist()}")
        return

    if args.mode == "source":
        ref = np.load("results/tt_full.npz")
        c0, a = ref["c0"], ref["a"]
        _, _, c_k = run_tt(args.k, c_init=torch.from_numpy(c0),
                           a_fixed=torch.from_numpy(a))
        np.savez(args.out, c=c_k, a=a, k=args.k, n=args.batches,
                 annotation=json.dumps({
                     "format": "device-neutral/ctx-state-v1",
                     "source_device": "tenstorrent-blackhole (tt-sim v1.16.0)",
                     "kernel": "vecacc",
                     "safe_point": "batch boundary",
                     "batches_done": args.k,
                     "live_values": ["C", "A", "batch_counter"],
                     "dtype": "int32",
                 }))
        print(f"source: {args.k}/{args.batches} batches captured, "
              f"state {os.path.getsize(args.out)} B -> {args.out}")
        return

    if args.mode == "verify":
        ref = np.load("results/tt_full.npz")
        st = np.load(args.out)
        mem = np.fromfile(args.spatial_mem, dtype=np.int32)
        c_spatial = mem[0:ELEMS]
        ok_ref = np.array_equal(c_spatial, ref["c_full"].reshape(-1))
        k, n = int(st["k"]), int(st["n"])
        c_model = st["c"][0, :] + (n - k) * st["a"][0, :]
        ok_model = np.array_equal(c_spatial, c_model.astype(np.int32))
        print(f"spatial C[0:4] = {c_spatial[:4].tolist()}")
        print(f"TT-full  C[0:4] = {ref['c_full'][0, :4].tolist()}")
        print("bit_identical_vs_tt_full:", bool(ok_ref))
        print("matches_python_model:    ", bool(ok_model))
        import sys
        sys.exit(0 if (ok_ref and ok_model) else 1)


if __name__ == "__main__":
    main()
