#!/bin/bash
# TT->TT migration matrix: 3 kernels x {baseline, instrumented, source->target} x 3 reps.
set -uo pipefail
cd /work/migration_tt
export TT_METAL_SIMULATOR=/work/ttsim/src/_out/release_bh/libttsim.so
export TT_METAL_SLOW_DISPATCH_MODE=1
PY=/opt/venv/bin/python
N=${N_BATCHES:-20}; K=${K_BATCH:-10}
run() { $PY migration_ops.py "$@" 2>&1 | grep -E '^\{|Error|error|Traceback|RuntimeError|assert' | tail -2; }
for rep in 0 1 2; do
  for kern in matmul vecadd pi; do
    echo "=== rep$rep $kern baseline ===";  run baseline     --kernel $kern --batches $N --rep $rep --out results/ops/base_${kern}_$rep.json
    echo "=== rep$rep $kern instrumented ==="; run instrumented --kernel $kern --batches $N --rep $rep --out results/ops/inst_${kern}_$rep.json
    echo "=== rep$rep $kern migrate ==="
    rm -f results/ops/state_${kern}_$rep.pt
    run source --kernel $kern --batches $N --k $K --rep $rep --state results/ops/state_${kern}_$rep.pt --out results/ops/src_${kern}_$rep.json
    run target --kernel $kern --batches $N --k $K --rep $rep --state results/ops/state_${kern}_$rep.pt --out results/ops/tgt_${kern}_$rep.json
  done
done
echo MATRIX_DONE
