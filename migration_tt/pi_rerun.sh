#!/bin/bash
cd /work/migration_tt
export TT_METAL_SIMULATOR=/work/ttsim/src/_out/release_bh/libttsim.so
export TT_METAL_SLOW_DISPATCH_MODE=1
PY=/opt/venv/bin/python
N=20; K=10
run() { $PY migration_ops.py "$@" 2>&1 | grep -E '^\{|Error|Traceback' | tail -1; }
for rep in 0 1 2; do
  rm -f results/ops/state_pi_$rep.pt
  run baseline     --kernel pi --batches $N --rep $rep --out results/ops/base_pi_$rep.json
  run instrumented --kernel pi --batches $N --rep $rep --out results/ops/inst_pi_$rep.json
  run source --kernel pi --batches $N --k $K --rep $rep --state results/ops/state_pi_$rep.pt --out results/ops/src_pi_$rep.json
  run target --kernel pi --batches $N --k $K --rep $rep --state results/ops/state_pi_$rep.pt --out results/ops/tgt_pi_$rep.json
done
echo PI_DONE
