# Heterogeneous XPU live-migration experiments

Companion artifacts for the paper's evaluation of checkpoint + migration
across backends. Three experiments, all driven through the same
device-neutral state protocol (`device-neutral/ctx-state-v1`): capture at a
safe point, serialize, restore on a fresh instance, resume, verify.

## 1. `migration_tt/` — TT→TT on tt-sim (three kernels, bit-identical)

Monte Carlo π, iterative matmul, and vector-add accumulate, each migrated
between two simulated Tenstorrent Blackhole devices (tt-sim v1.16.0 +
TT-Metalium v0.81.1). Migration is cross-process because `libttsim.so` is a
process singleton — the target is a genuinely fresh device instance.

- Harness: `migration_ops.py` (modes: `baseline`, `instrumented`,
  `source`, `target`, `calibrate`), runner `run_ops_experiment.sh`,
  summary `summarize_ops.py` (includes the bit-identity verification).
- RISC-V dataflow kernels for the kernel-level variant: `pi_kernel.c`,
  `matmul_kernel.c`, `vecadd_kernel.c`.
- Results: `results/ops/*.json` (3 kernels × 4 modes × 3 reps),
  `results/ops/summary.json`.
- Measured (20 batches, per-batch instrumentation enabled):
  matmul +0.31% end-to-end / vecadd +33% / pi +4.0% vs uninterrupted runs;
  all migrated runs bit-identical.
- Environment: docker `ubuntu:24.04`, venv with the `ttnn` wheel (v0.81.1),
  `TT_METAL_SIMULATOR=<libttsim_bh.so> TT_METAL_SLOW_DISPATCH_MODE=1`, plus
  a `soc_descriptor.yaml` copied next to the simulator `.so` and SFPI at
  `/opt/tenstorrent/sfpi`.

## 2. `spatial_e2e/` — spatial lowering + end-to-end kernel + live migration

The full pipeline the paper describes, executable end to end on one kernel:

```
kernel PTX --ptx_to_ctx.py--> CTX muops --ctx_lower.py--> ROM --> spatial_core.sv (Verilator) --oracle.py--> bit/cycle-exact check
```

- `ptx_to_ctx.py`: lifts the supported PTX fragment into CTX — lane-packed
  guards on every muop, state spaces and order/scope labels on memory,
  short guarded diamonds if-converted into mask split/merge pairs,
  `bar.sync` muops labeled as safe suspension points.
- `ctx_lower.py`: region formation, stream plans (base / per-lane address /
  lane count), the ordered atomic as a serialized token island, identity
  schedule → ROM.
- `spatial_core.sv` + `tb_main.cpp`: 32-lane lane-packed RTL core with
  mask-guarded updates, per-lane memory-event accounting, token islands,
  full architectural state export/import. Verilator 5.028.
- `oracle.py`: bit-exact software model; checks memory AND cycle/event/
  token-wait counts.
- `run_spatial.sh`: 3 configurations × {baseline, migrate, resume}.
- Results (`results/`): every configuration matches the oracle on memory,
  cycles, events, and token waits; migration at the post-barrier safe point
  (PC 41, 266 KB state) into a freshly elaborated Verilator instance in a
  new process loses zero cycles (180+1179 = 1359) and is memory
  bit-identical.
- `run_tt_to_spatial.sh` + `tt_to_spatial.py`: TT→spatial feasibility —
  10 batches of an int32 vector-add accumulation on TT-sim, device-neutral
  checkpoint, the spatial core resumes the remaining 10 batches;
  bit-identical to the uninterrupted TT run.

## 3. `qwen_nv_tt/` — NVIDIA→TT with a Qwen-0.6B layer + DWARF→CTX recovery

- `qwen_nv.py`: Qwen3-0.6B decoder layer 0 (real weights, bf16, eager
  attention, growing KV cache) running autoregressive decode on a real GPU
  (RTX PRO 6000 Blackwell). `full` = uninterrupted reference; `source` =
  checkpoint at step K (hidden state, KV cache, step counter, CTX
  annotation); `harvest` = torch.compile the step and collect the inductor
  PTX (`ptx_corpus/`, 8 real Triton kernels, sm_120a).
- `qwen_tt.py`: the same layer math in ttnn on tt-sim. `target` restores
  the NVIDIA checkpoint (bit-exact: the restored state hashes to the
  captured SHA) and finishes steps 6..11.
- Results (`results/qwen_nvtt_result.json`): cross-backend agreement
  max-abs 0.1875 / rel 1.3% / cosine 0.99997 vs the NVIDIA-only run; the
  TT-only 12-step run differs by 2.1% — migration deviation is inside
  ordinary backend arithmetic variation.
- `dwarf_to_ctx.py`: DWARF → CTX state recovery. Compiles a captured kernel
  PTX with `ptxas -g`, masks the cubin's non-standard RELA section headers
  (the names/lines/types are relocation-independent), and walks
  `.debug_info`/`.debug_line`: compilation units, `DW_TAG_subprogram` PC
  ranges, variable DIEs (name / decl line / type / location), and the
  address→line table. Emits the CTX state annotation the checkpoint ships
  (`results/ctx_annotation_k*.json`). Rationale: per-variable DWARF
  locations are ephemeral on optimized device code, and reconstructing
  target-side DWARF through a different ISA and register allocator is the
  hard problem — CTX's unbounded virtual registers give the checkpoint a
  stable state space instead, so DWARF is consumed only on the NVIDIA side
  as provenance and re-expressed as CTX annotations.

## Environment notes

- tt-sim: `./make.py :build` (or the v1.16.0 release `.so`).
- The `tt-metalium` GitHub-release debs are broken (the base package is not
  published); use `pip install ttnn` (bundled runtime) or build tt-metal
  from source (done against v0.81.1, gcc-14 toolchain file).
- Weights for experiment 3: `Qwen/Qwen3-0.6B` from HuggingFace (copy the
  safetensors into the directory; the loader accepts a file path via
  `QWEN_SNAP`).
