#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${HETGPU_WASM_TARGET:-wasm32-unknown-emscripten}"
profile="${HETGPU_WASM_PROFILE:-release}"
out="${1:-$repo_root/cxl/lib/libhetgpu_cuda_wasm.a}"
features="${HETGPU_WASM_FEATURES:-webgpu,embed_cudart}"

if ! rustup target list --installed | grep -qx "$target"; then
  echo "missing Rust target: $target" >&2
  echo "install it with: rustup target add $target" >&2
  exit 1
fi

mkdir -p "$(dirname "$out")"

args=(rustc -p zluda --target "$target" --no-default-features --features "$features" --lib --crate-type staticlib)
if [[ "$profile" == "release" ]]; then
  args+=(--release)
fi

(
  cd "$repo_root"
  HETGPU_EMBED_CUDART_STATIC=1 cargo "${args[@]}"
)

archive="$repo_root/target/$target/$profile/libnvcuda.a"
if [[ ! -f "$archive" ]]; then
  echo "expected archive was not produced: $archive" >&2
  exit 1
fi

cp "$archive" "$out"
echo "$out"
