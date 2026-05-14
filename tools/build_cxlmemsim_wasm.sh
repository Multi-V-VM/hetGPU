#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cxlmemsim_root="${CXLMEMSIM_ROOT:-$repo_root/CXLMemSim}"
deploy_dir="${1:-$repo_root/victoryang00.github.io/cxl2/cxlmemsim_wasm}"
build_dir="${CXLMEMSIM_WASM_BUILD_DIR:-$cxlmemsim_root/build-wasm}"

if ! command -v emcmake >/dev/null 2>&1; then
  echo "emcmake not found; source emsdk_env.sh before running this" >&2
  exit 1
fi

mkdir -p "$deploy_dir"

emcmake cmake -S "$cxlmemsim_root" -B "$build_dir" \
  -DCXLMEMSIM_BUILD_WASM=ON \
  -DCXLMEMSIM_BUILD_MICROBENCHMARKS=OFF \
  -DCXLMEMSIM_ENABLE_RDMA=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$build_dir" --target cxlmemsim_wasm

cp "$build_dir/cxlmemsim_wasm.mjs"  "$deploy_dir/cxlmemsim_wasm.mjs"
cp "$build_dir/cxlmemsim_wasm.wasm" "$deploy_dir/cxlmemsim_wasm.wasm"

echo "$deploy_dir/cxlmemsim_wasm.mjs"
echo "$deploy_dir/cxlmemsim_wasm.wasm"
