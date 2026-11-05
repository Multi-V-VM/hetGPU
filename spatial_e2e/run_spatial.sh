#!/bin/bash
# Spatial end-to-end + live migration benchmark: 3 configs x {baseline, migrate, resume}.
set -e
cd /root/tt-eval/spatial_e2e
mkdir -p results
for cfg in "8 20 20261105" "16 13 314159" "4 31 999331"; do
  set -- $cfg; TILES=$1; TAILW=$2; SEED=$3
  echo "=== tiles=$TILES tailw=$TAILW seed=$SEED ==="
  ./obj_dir/spatial_run --mode run --rom rom_mm.hex --tiles $TILES --tailw $TAILW --seed $SEED \
     --out results/base_${TILES}_${TAILW}.json | tail -1
  python3 oracle.py check --rom rom_mm.hex --mem results/base_${TILES}_${TAILW}.json.mem --tiles $TILES --tailw $TAILW --seed $SEED | tail -2
  rm -f results/state_${TILES}_${TAILW}.bin
  ./obj_dir/spatial_run --mode migrate --rom rom_mm.hex --tiles $TILES --tailw $TAILW --seed $SEED --at-pc 41 \
     --state results/state_${TILES}_${TAILW}.bin --out results/src_${TILES}_${TAILW}.json | tail -1
  ./obj_dir/spatial_run --mode resume --rom rom_mm.hex --tiles $TILES --tailw $TAILW --seed $SEED \
     --state results/state_${TILES}_${TAILW}.bin --out results/tgt_${TILES}_${TAILW}.json | tail -1
  cmp -s results/base_${TILES}_${TAILW}.json.mem results/tgt_${TILES}_${TAILW}.json.mem && echo "MEM_BITIDENTICAL" || echo "MEM_MISMATCH"
done
echo SPATIAL_DONE
