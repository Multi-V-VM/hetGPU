#!/usr/bin/env python
# NV stage of the NV -> TT -> spatial chain: run the max-gated accumulate
# workload C = max(C + A, B) (int32, 4096 elements) on the real GPU.
#
#   chain_nv.py full   --batches N               -> uninterrupted reference
#   chain_nv.py source --k K1 --batches N --out  -> checkpoint after K1 batches
#
# int32 add/max are exact on every backend, so the whole chain verifies
# bit-exact end to end.

import argparse
import hashlib
import json
import os

import numpy as np
import torch

SEED = 0xC0FFEE
ELEMS = 4096


def init():
    g = torch.Generator().manual_seed(SEED)
    c = torch.randint(0, 1000, (1, ELEMS), generator=g, dtype=torch.int32)
    a = torch.randint(1, 500, (1, ELEMS), generator=g, dtype=torch.int32)
    b = torch.randint(500, 3000, (1, ELEMS), generator=g, dtype=torch.int32)
    return c, a, b


def batches(c, a, b, n):
    for _ in range(n):
        c = torch.maximum(c + a, b)
    return c


def save_state(path, c, a, b, k, n):
    np.savez(path, c=c.numpy(), a=a.numpy(), b=b.numpy(),
             k=np.array(k), n=np.array(n))
    sha = hashlib.sha256(c.numpy().tobytes()).hexdigest()[:16]
    ann = {"format": "device-neutral/ctx-state-v1",
           "source_device": "nvidia-rtx-pro-6000-blackwell (sm_120, CUDA 13.2)",
           "workload": "vecmaxacc: C = max(C+A, B), int32, 4096 elements",
           "safe_point": "batch boundary",
           "batches_done": k, "state_sha": sha,
           "live_values": ["C", "A", "B", "batch_counter"]}
    json.dump(ann, open(path + ".ctx.json", "w"), indent=1)
    return sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["full", "source"])
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="results/chain_state_nv.npz")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    c0, a, b = init()
    if args.mode == "full":
        c = batches(c0, a, b, args.batches)
        np.savez("results/chain_full.npz", c_full=c.numpy())
        print(f"NV full: {args.batches} batches, C[0:4]={c[0, :4].tolist()}")
        return

    c = batches(c0, a, b, args.k)
    sha = save_state(args.out, c, a, b, args.k, args.batches)
    print(f"NV source: {args.k}/{args.batches} batches, sha {sha} -> {args.out}")


if __name__ == "__main__":
    main()
