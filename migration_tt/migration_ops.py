#!/usr/bin/env python
# TT->TT live migration on simulated Blackhole (tt-sim + TT-Metalium/ttnn).
#
# Mechanism: the paper's segmented-kernel scheme (sec 5.2): "A global barrier
# every X iterations turns a long-running kernel into a chain of standard
# launches whose boundary state is exactly a snapshot." Each batch is one
# launch of the workload's op sequence; the runtime captures the device
# working set at the boundary and, in a migration, restores it on a freshly
# created device in a new process (libttsim is a process singleton, so
# cross-process is the only honest fresh-device boundary).
#
# Modes:
#   baseline     N batches, no boundary captures
#   instrumented N batches, capture (device->host readback) at every boundary
#   source       batches [0,K), capture every boundary, serialize state, exit
#   target       restore state on a fresh device, batches [K,N)
#   calibrate    1 batch, print timing
#
# Workloads (device-resident state evolves per batch):
#   matmul  X = X @ B    (compute-bound; state = 256x256 bf16 tile)
#   vecadd  C = C + A    (bandwidth-bound; state = 1M bf16 elements)
#   pi      hits += sum(x^2+y^2 < 1) over a point tile per batch
#           (device state = hits accumulator; point stream drawn host-side,
#            the device does the FP numerics)
#
# All timing uses CLOCK_REALTIME nanoseconds so cross-process components
# compose: the source stamps capture_start_ns before its final capture and
# the target stamps first_resume_done_ns after its first post-restore batch.

import argparse
import json
import os
import time

import torch
import ttnn


def ns():
    return time.time_ns()


class Timer:
    def __init__(self):
        self.phases = {}

    def add(self, name, t0):
        self.phases[name] = self.phases.get(name, 0.0) + (time.time_ns() - t0) / 1e9

    def add_s(self, name, seconds):
        self.phases[name] = self.phases.get(name, 0.0) + seconds

    def to_dict(self):
        return {k: round(v, 4) for k, v in self.phases.items()}


def to_device(dev, cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    t0 = ns()
    t = ttnn.from_torch(cpu, dtype=dtype, layout=layout, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return t, (ns() - t0) / 1e9


def to_host(t):
    t0 = ns()
    cpu = ttnn.to_torch(t)
    return cpu, (ns() - t0) / 1e9


class Workload:
    name = "?"

    def __init__(self, dev, seed=0xC0FFEE):
        self.dev = dev
        self.seed = seed
        self._g = torch.Generator().manual_seed(seed)
        t0 = ns()
        self.build()
        self.build_s = (ns() - t0) / 1e9

    def build(self):
        raise NotImplementedError

    def batch(self):
        raise NotImplementedError

    def capture(self):
        """Return (snapshot dict, capture_seconds)."""
        raise NotImplementedError

    def restore(self, snap):
        """Return restore_seconds."""
        raise NotImplementedError

    def save_final(self, snap, path):
        raise NotImplementedError

    def verify_output(self, snap):
        raise NotImplementedError


class Matmul(Workload):
    """X_{k+1} = X_k @ B (uint-free bf16 flow); the paper's migration
    workload, 256x256 per batch."""
    name = "matmul"
    N = 512

    def build(self):
        n = self.N
        x = torch.randn(n, n, generator=self._g).to(torch.bfloat16)
        b = torch.randn(n, n, generator=self._g).to(torch.bfloat16)
        self.tx, _ = to_device(self.dev, x)
        self.tb, _ = to_device(self.dev, b)

    def batch(self):
        t0 = ns()
        self.tx = ttnn.matmul(self.tx, self.tb,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.batch_s = (ns() - t0) / 1e9

    def capture(self):
        cpu, dt = to_host(self.tx)
        return {"x": cpu}, dt

    def restore(self, snap):
        self.tx, dt = to_device(self.dev, snap["x"])
        return dt

    def save_final(self, snap, path):
        torch.save({"x": snap["x"]}, path)

    def verify_output(self, snap):
        return {"checksum": round(float(snap["x"].sum().item()), 4)}


class VecAdd(Workload):
    """C = C + A on 1M bf16 elements; bandwidth-bound contrast."""
    name = "vecadd"
    ELEMS = 1 << 19

    def build(self):
        n = self.ELEMS
        c = torch.randn(1, n, generator=self._g).to(torch.bfloat16)
        a = torch.randn(1, n, generator=self._g).to(torch.bfloat16)
        self.tc, _ = to_device(self.dev, c)
        self.ta, _ = to_device(self.dev, a)

    def batch(self):
        t0 = ns()
        self.tc = ttnn.add(self.tc, self.ta,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.batch_s = (ns() - t0) / 1e9

    def capture(self):
        cpu, dt = to_host(self.tc)
        return {"c": cpu}, dt

    def restore(self, snap):
        self.tc, dt = to_device(self.dev, snap["c"])
        return dt

    def save_final(self, snap, path):
        torch.save({"c": snap["c"]}, path)

    def verify_output(self, snap):
        return {"checksum": round(float(snap["c"].sum().item()), 4)}


class Pi(Workload):
    """Monte Carlo pi: per batch the device computes sum(x^2+y^2<1) over a
    64K-point tile and accumulates into a device-resident hits scalar. The
    point stream is drawn host-side from a seeded generator; the device does
    the FP numerics and holds the evolving accumulator state."""
    name = "pi"
    POINTS = 1 << 11

    def build(self):
        p = self.POINTS
        self.hits_cpu = torch.zeros(1, dtype=torch.float32)
        self.count = 0
        self.thits, _ = to_device(self.dev, self.hits_cpu,
                                  dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def batch(self):
        p = self.POINTS
        t0 = ns()
        xy = torch.rand(1, 2 * p, generator=self._g).to(torch.float32)
        tx, _ = to_device(self.dev, xy, dtype=ttnn.float32,
                          layout=ttnn.ROW_MAJOR_LAYOUT)
        self.upload_s = getattr(self, "upload_s", 0.0) + (ns() - t0) / 1e9
        x = ttnn.slice(tx, ttnn.Shape([0, 0]), ttnn.Shape([1, p]),
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
        y = ttnn.slice(tx, ttnn.Shape([0, p]), ttnn.Shape([1, 2 * p]),
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
        r = ttnn.add(ttnn.multiply(x, x, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                     ttnn.multiply(y, y, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
        inside = ttnn.lt(r, 1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        bhits = ttnn.sum(ttnn.typecast(inside, ttnn.float32),
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.thits = ttnn.add(self.thits,
                              ttnn.reshape(bhits, ttnn.Shape([1])),
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.count += p
        self.batch_s = (ns() - t0) / 1e9

    def capture(self):
        cpu, dt = to_host(self.thits)
        return {"hits": cpu.reshape(1), "count": torch.tensor(float(self.count))}, dt

    def restore(self, snap):
        self.count = int(snap["count"].item())
        self.thits, dt = to_device(self.dev, snap["hits"].reshape(1),
                                   dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
        return dt

    def save_final(self, snap, path):
        torch.save({"hits": snap["hits"], "count": snap["count"]}, path)

    def verify_output(self, snap):
        h = float(snap["hits"].item())
        return {"hits": h, "pi_est": round(4.0 * h / max(self.count, 1), 6)}


WORKLOADS = {w.name: w for w in (Matmul, VecAdd, Pi)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["baseline", "instrumented", "source",
                                     "target", "calibrate"])
    ap.add_argument("--kernel", default="matmul", choices=list(WORKLOADS))
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--state", default="results/state.pt")
    ap.add_argument("--final", default=None, help="final-tensor .pt path")
    ap.add_argument("--out", default=None, help="JSON results path")
    ap.add_argument("--rep", type=int, default=0)
    args = ap.parse_args()

    for p in (os.path.dirname(args.out or "") , os.path.dirname(args.state),
              os.path.dirname(args.final or "")):
        if p:
            os.makedirs(p, exist_ok=True)

    res = {"mode": args.mode, "kernel": args.kernel, "rep": args.rep,
           "n_batches": args.batches, "k": args.k}

    t0 = ns()
    dev = ttnn.CreateDevice(device_id=0)
    res["create_device_s"] = round((ns() - t0) / 1e9, 4)

    w = WORKLOADS[args.kernel](dev)
    res["build_s"] = round(w.build_s, 4)
    T = Timer()

    if args.mode == "calibrate":
        t0 = ns()
        w.batch()
        one = (ns() - t0) / 1e9
        t0 = ns()
        w.capture()
        cap = (ns() - t0) / 1e9
        ttnn.CloseDevice(dev)
        print(json.dumps({"kernel": args.kernel, "one_batch_s": round(one, 4),
                          "build_s": round(w.build_s, 4),
                          "capture_s": round(cap, 4)}))
        return

    if args.mode == "baseline":
        for i in range(args.batches):
            t0 = ns()
            w.batch()
            T.add("batch_exec", t0)
            if i == 0:
                res["first_batch_s"] = round(w.batch_s, 4)
        snap, dt = w.capture()
        T.add_s("final_capture", dt)

    elif args.mode == "instrumented":
        snap = None
        for i in range(args.batches):
            t0 = ns()
            w.batch()
            T.add("batch_exec", t0)
            if i == 0:
                res["first_batch_s"] = round(w.batch_s, 4)
            snap, dt = w.capture()
            T.add_s("capture", dt)

    elif args.mode == "source":
        snap = None
        for i in range(args.k):
            t0 = ns()
            w.batch()
            T.add("batch_exec", t0)
            snap, dt = w.capture()
            T.add_s("capture", dt)
        res["capture_start_ns"] = ns()
        t0 = ns()
        torch.save({"kernel": args.kernel, "batches_done": args.k,
                    "snap": snap, "seed": w.seed,
                    "rng_state": w._g.get_state(),
                    "capture_start_ns": res["capture_start_ns"]},
                   args.state)
        T.add_s("serialize", (ns() - t0) / 1e9)
        res["state_file"] = args.state
        res["state_bytes"] = os.path.getsize(args.state)

    elif args.mode == "target":
        t0 = ns()
        payload = torch.load(args.state, weights_only=False)
        T.add_s("deserialize", (ns() - t0) / 1e9)
        res["state_bytes"] = os.path.getsize(args.state)
        if "rng_state" in payload:
            w._g.set_state(payload["rng_state"])
        T.add_s("restore", w.restore(payload["snap"]))
        first_done = None
        for i in range(payload["batches_done"], args.batches):
            t0 = ns()
            w.batch()
            T.add("batch_exec", t0)
            if first_done is None:
                first_done = ns()
                res["first_resume_done_ns"] = first_done
                res["first_resume_batch_s"] = round(w.batch_s, 4)
            snap, dt = w.capture()
            T.add_s("capture", dt)
        res["capture_start_ns"] = payload["capture_start_ns"]

    if args.mode in ("baseline", "instrumented", "target"):
        res.update(w.verify_output(snap))
        final = args.final or f"results/final_{args.mode}_{args.kernel}_{args.rep}.pt"
        w.save_final(snap, final)
        res["final_file"] = final
    if isinstance(w, Pi) and getattr(w, "upload_s", None):
        res["point_upload_s"] = round(w.upload_s, 4)

    t0 = ns()
    ttnn.CloseDevice(dev)
    res["close_device_s"] = round((ns() - t0) / 1e9, 4)
    res["timings"] = T.to_dict()
    res["total_s"] = round(sum(T.phases.values()), 4)

    out = args.out or f"results/{args.mode}_{args.kernel}_{args.rep}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
