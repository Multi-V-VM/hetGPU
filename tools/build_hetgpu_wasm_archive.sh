#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${HETGPU_WASM_TARGET:-wasm32-unknown-emscripten}"
profile="${HETGPU_WASM_PROFILE:-release}"
out="${1:-$repo_root/cxl/lib/libhetgpu_cuda_wasm.a}"
features="${HETGPU_WASM_FEATURES:-webgpu,embed_cudart}"
build_std="${HETGPU_WASM_BUILD_STD:-1}"
cargo_home="${CARGO_HOME:-$repo_root/qemu/.cache/cargo}"
rustflags="${RUSTFLAGS:-}"
target_cflags="${TARGET_CFLAGS:-}"
target_cflags_underscored="${CFLAGS_wasm32_unknown_emscripten:-}"

if ! rustup target list --installed | grep -qx "$target"; then
  echo "missing Rust target: $target" >&2
  echo "install it with: rustup target add $target" >&2
  exit 1
fi

mkdir -p "$(dirname "$out")"
mkdir -p "$cargo_home"

args=(rustc -p zluda --target "$target" --no-default-features --features "$features" --lib --crate-type staticlib)
if [[ "$profile" == "release" ]]; then
  args+=(--release)
fi
cargo_cmd=(cargo)
if [[ "$build_std" != "0" ]]; then
  cargo_cmd+=(-Z build-std=std,panic_abort)
fi

(
  cd "$repo_root"
  export CARGO_HOME="$cargo_home"
  if [[ "$build_std" != "0" ]]; then
    export RUSTC_BOOTSTRAP="${RUSTC_BOOTSTRAP:-1}"
  fi
  RUSTFLAGS="${rustflags:+$rustflags }-C target-feature=+atomics,+bulk-memory -C panic=abort -C link-arg=-pthread" \
  TARGET_CFLAGS="${target_cflags:+$target_cflags }-pthread -matomics -mbulk-memory" \
  CFLAGS_wasm32_unknown_emscripten="${target_cflags_underscored:+$target_cflags_underscored }-pthread -matomics -mbulk-memory" \
  HETGPU_EMBED_CUDART_STATIC=1 \
  "${cargo_cmd[@]}" "${args[@]}"
)

archive="$repo_root/target/$target/$profile/libnvcuda.a"
if [[ ! -f "$archive" ]]; then
  echo "expected archive was not produced: $archive" >&2
  exit 1
fi

cp "$archive" "$out"
echo "$out"
