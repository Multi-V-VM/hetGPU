#!/bin/bash
# TT->spatial feasibility: K batches on TT-sim -> neutral state -> spatial
# core executes the remaining batches -> bit-identity vs full TT run.
set -e
cd /root/tt-eval/spatial_e2e
export TT_METAL_SIMULATOR=/work/ttsim/src/_out/release_bh/libttsim.so
export TT_METAL_SLOW_DISPATCH_MODE=1
N=${N:-20}; K=${K:-10}
PY=/opt/venv/bin/python
PYHOST=python3

echo "=== TT full reference ($N batches) ==="
docker exec -w /work/spatial_e2e -e TT_METAL_SIMULATOR=/work/ttsim/src/_out/release_bh/libttsim.so -e TT_METAL_SLOW_DISPATCH_MODE=1 tt-mig /opt/venv/bin/python /work/spatial_e2e/tt_to_spatial.py full --batches $N | tail -1
echo "=== TT source stage ($K batches -> checkpoint) ==="
docker exec -w /work/spatial_e2e -e TT_METAL_SIMULATOR=/work/ttsim/src/_out/release_bh/libttsim.so -e TT_METAL_SLOW_DISPATCH_MODE=1 tt-mig /opt/venv/bin/python /work/spatial_e2e/tt_to_spatial.py source --k $K --batches $N --out results/tt_state.npz | tail -1

echo "=== bridge: neutral state -> spatial memory image ==="
$PYHOST - <<'PY'
import numpy as np
st = np.load("results/tt_state.npz")
mem = np.zeros(1 << 16, dtype=np.int32)
mem[0:4096] = st["c"][0, :]          # C at base 0
mem[8192:8192 + 4096] = st["a"][0, :]  # A at base 8192
mem.tofile("results/bridge_mem.bin")
open("results/bridge_params.txt", "w").write(f"0 8192 {int(st['n']) - int(st['k'])} 0 0")
print(f"bridge: C and A placed, {int(st['n'])-int(st['k'])} remaining batches")
PY

echo "=== spatial stage: resume on RTL core ==="
read -r P0 P1 P2 P3 P4 < results/bridge_params.txt
./obj_dir/spatial_run --mode run --rom rom_vecacc.hex --max-cycles 200000 \
  --params $P0 $P1 $P2 $P3 $P4 --mem-init results/bridge_mem.bin \
  --out results/spatial_tgt.json | tail -2

echo "=== verify ==="
$PYHOST tt_to_spatial.py verify --spatial-mem results/spatial_tgt.json.mem
