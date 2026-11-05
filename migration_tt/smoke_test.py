# Smoke test: ttnn on simulated Blackhole. Create device, run one add,
# read back, close. Run with TT_METAL_SIMULATOR + TT_METAL_SLOW_DISPATCH_MODE.
import os, sys, time

import ttnn
import torch

print("ttnn:", ttnn.__file__)
print("has random:", hasattr(ttnn, "random"))

t0 = time.time()
dev = ttnn.CreateDevice(device_id=0)
print(f"device created in {time.time()-t0:.2f}s; arch:", dev.arch(), flush=True)

a = torch.randn(256, 256, dtype=torch.float32)
b = torch.randn(256, 256, dtype=torch.float32)
ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
tc = ttnn.add(ta, tb, memory_config=ttnn.DRAM_MEMORY_CONFIG)
out = ttnn.to_torch(tc)
ref = (a.float() + b.float()).to(torch.bfloat16).float()
print("max abs diff vs cpu ref:", (out - ref).abs().max().item())

t0 = time.time()
tm = ttnn.matmul(ta, tb, memory_config=ttnn.DRAM_MEMORY_CONFIG)
m = ttnn.to_torch(tm)
print(f"matmul ok {m.shape} in {time.time()-t0:.2f}s")

ttnn.CloseDevice(dev)
print("SMOKE PASS")
