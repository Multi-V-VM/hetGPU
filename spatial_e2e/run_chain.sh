#!/bin/bash
# NV -> TT -> spatial one-stop chain: C = max(C+A, B), int32.
#   NV (real RTX 6000): 10 batches  -> checkpoint
#   TT (tt-sim BH):    +10 batches -> checkpoint
#   spatial (Verilator): +10 batches -> final
# Verified bit-exact against the uninterrupted NV run of all 30 batches.
set -e
cd /root/tt-eval/spatial_e2e
export TT_METAL_SIMULATOR=/work/ttsim/src/_out/release_bh/libttsim.so
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
PY=/opt/venv/bin/python
N=30; K1=10; K2=10

echo "=== [1] NV stage: full reference ($N batches, real GPU) ==="
python3 chain_nv.py full --batches $N | tail -1

echo "=== [2] NV source stage ($K1 batches -> checkpoint) ==="
python3 chain_nv.py source --k $K1 --batches $N | tail -1

echo "=== [3] TT stage (+$K2 batches on tt-sim -> checkpoint) ==="
docker cp /work/../../root/tt-eval/qwen_nv_tt/results/chain_state_nv.npz tt-mig:/work/spatial_e2e/results/ 2>/dev/null || \
  docker cp /root/tt-eval/spatial_e2e/results/chain_state_nv.npz tt-mig:/work/spatial_e2e/results/
docker cp chain_tt.py tt-mig:/work/spatial_e2e/
docker exec -w /work/spatial_e2e \
  -e TT_METAL_SIMULATOR=/work/ttsim/src/_out/release_bh/libttsim.so \
  -e TT_METAL_SLOW_DISPATCH_MODE=1 -e TT_METAL_DISABLE_SFPLOADMACRO=1 \
  tt-mig $PY chain_tt.py --state results/chain_state_nv.npz --k2 $K2 | tail -1
docker cp tt-mig:/work/spatial_e2e/results/chain_state_tt.npz results/

echo "=== [4] bridge: TT state -> spatial memory image ==="
python3 - <<'PY'
import numpy as np
st = np.load("results/chain_state_tt.npz")
mem = np.zeros(1 << 16, dtype=np.int32)
mem[0:4096] = st["c"][0, :]
mem[8192:8192+4096] = st["a"][0, :]
mem[12288:12288+4096] = st["b"][0, :]
mem.tofile("results/chain_mem.bin")
nb = int(st["n"]) - int(st["k"])
open("results/chain_params.txt", "w").write(f"0 8192 12288 {nb} 0")
print(f"bridge: C/A/B placed, {nb} remaining batches")
PY

echo "=== [5] spatial stage (fresh RTL instance, +${N}-${K1}-${K2} batches) ==="
read -r P0 P1 P2 P3 P4 < results/chain_params.txt
./obj_dir/spatial_run --mode run --rom rom_vecmaxacc.hex --max-cycles 400000 \
  --params $P0 $P1 $P2 $P3 $P4 --mem-init results/chain_mem.bin \
  --out results/chain_spatial.json | tail -1
docker cp results/chain_spatial.json.mem tt-mig:/work/spatial_e2e/results/ 2>/dev/null || cp results/chain_spatial.json.mem /root/tt-eval/spatial_e2e/results/

echo "=== [6] verify: bit-exact against the uninterrupted NV run ==="
python3 - <<'PY'
import numpy as np
ref = np.load("results/chain_full.npz")["c_full"].reshape(-1)
st = np.load("results/chain_state_tt.npz")
mem = np.fromfile("results/chain_spatial.json.mem", dtype=np.int32)
c_sp = mem[0:4096]
ok_ref = np.array_equal(c_sp, ref)
c_model = st["c"][0, :] + (int(st["n"]) - int(st["k"])) * 0  # continued by spatial
# python model for the spatial segment: continue from the TT state
c = st["c"][0, :].copy(); a = st["a"][0, :]; b = st["b"][0, :]
for _ in range(int(st["n"]) - int(st["k"])):
    c = np.maximum(c + a, b)
ok_model = np.array_equal(c_sp, c.astype(np.int32))
print("spatial C[0:4]     =", c_sp[:4].tolist())
print("NV-full C[0:4]     =", ref[:4].tolist())
print("bit_identical_vs_NV_full:", bool(ok_ref))
print("matches_python_model:    :", bool(ok_model))
import sys; sys.exit(0 if (ok_ref and ok_model) else 1)
PY
echo CHAIN_DONE
