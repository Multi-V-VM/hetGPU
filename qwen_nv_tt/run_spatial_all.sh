#!/bin/bash
# All spatial kernels: baseline + oracle match.
set -e
cd /root/tt-eval/spatial_e2e
mkdir -p results
run_kernel() {
  local rom=$1 tiles=$2 tailw=$3 seed=$4 name=$5
  ./obj_dir/spatial_run --mode run --rom $rom --tiles $tiles --tailw $tailw --seed $seed \
     --out results/${name}_base.json | tail -1
  python3 oracle.py check --rom $rom --mem results/${name}_base.json.mem --tiles $tiles --tailw $tailw --seed $seed | tail -1
}
run_kernel rom_lower.hex      8 20 20261105 mm_reduce_8x20
run_kernel rom_lower.hex     16 13 314159   mm_reduce_16x13
run_kernel rom_lower.hex      4 31 999331   mm_reduce_4x31
run_kernel rom_max_reduce.hex 1 1  20261105 max_reduce
run_kernel rom_histogram.hex  1 1  20261105 histogram
echo ALL_SPATIAL_DONE
