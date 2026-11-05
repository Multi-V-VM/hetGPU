#!/bin/bash
set -uo pipefail
cd /work/migration_tt
export TT_METAL_SIMULATOR=/work/ttsim/src/_out/release_bh/libttsim.so
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
PY=/opt/venv/bin/python
N=${N_BATCHES:-12}; K=${K_BATCH:-6}
run() { $PY migration_ops.py "$@" 2>&1 | grep -E '^\{|Error|Traceback' | tail -1; }
for rep in 0 1 2; do
  for kern in attn mlp normchain; do
    run baseline     --kernel $kern --batches $N --rep $rep --out results/ops/base_${kern}_$rep.json
    run instrumented --kernel $kern --batches $N --rep $rep --out results/ops/inst_${kern}_$rep.json
    rm -f results/ops/state_${kern}_$rep.pt
    run source --kernel $kern --batches $N --k $K --rep $rep --state results/ops/state_${kern}_$rep.pt --out results/ops/src_${kern}_$rep.json
    run target --kernel $kern --batches $N --k $K --rep $rep --state results/ops/state_${kern}_$rep.pt --out results/ops/tgt_${kern}_$rep.json
  done
done
echo NEW_KERNELS_DONE
