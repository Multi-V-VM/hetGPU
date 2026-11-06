#!/usr/bin/env python
# TT stage of the NV -> TT -> spatial chain: restore the NVIDIA checkpoint
# on a simulated Tenstorrent Blackhole, run the middle segment of batches
# (ttnn.add + ttnn.maximum, int32, bit-exact), and re-checkpoint for the
# spatial stage.
#
#   chain_tt.py --state results/chain_state_nv.npz --k2 K2

import argparse
import hashlib
import json
import os

import numpy as np
import torch
import ttnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="results/chain_state_nv.npz")
    ap.add_argument("--k2", type=int, default=10)
    args = ap.parse_args()

    st = np.load(args.state)
    sha_in = hashlib.sha256(st["c"].tobytes()).hexdigest()[:16]
    print(f"restored state sha {sha_in} (captured {st['k']} batches)")

    dev = ttnn.CreateDevice(device_id=0)
    tc = ttnn.from_torch(st["c"], dtype=ttnn.int32, device=dev,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ta = ttnn.from_torch(st["a"], dtype=ttnn.int32, device=dev,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tb = ttnn.from_torch(st["b"], dtype=ttnn.int32, device=dev,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for _ in range(args.k2):
        tc = ttnn.maximum(ttnn.add(tc, ta, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                          tb, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    c = ttnn.to_torch(tc).to(torch.int32)
    ttnn.CloseDevice(dev)

    k_total = int(st["k"]) + args.k2
    np.savez("results/chain_state_tt.npz", c=c.numpy(), a=st["a"], b=st["b"],
             k=np.array(k_total), n=st["n"])
    sha = hashlib.sha256(c.numpy().tobytes()).hexdigest()[:16]
    ann = {"format": "device-neutral/ctx-state-v1",
           "source_device": "tenstorrent-blackhole (tt-sim v1.16.0)",
           "workload": "vecmaxacc: C = max(C+A, B), int32, 4096 elements",
           "safe_point": "batch boundary",
           "batches_done": k_total, "state_sha": sha,
           "live_values": ["C", "A", "B", "batch_counter"]}
    json.dump(ann, open("results/chain_state_tt.npz.ctx.json", "w"),
              indent=1)
    print(f"TT stage: +{args.k2} batches (total {k_total}), "
          f"sha {sha} -> results/chain_state_tt.npz")


if __name__ == "__main__":
    main()
