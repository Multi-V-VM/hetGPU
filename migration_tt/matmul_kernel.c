// SPDX-License-Identifier: Apache-2.0
//
// Iterative int32 matmul worker kernel for the TT->TT live-migration
// experiment. Per batch: X = X * B (natural uint32 wraparound), where B is
// regenerated deterministically from the kernel seed. This mirrors the
// paper's migration workload (long-running iterative tiled matmul).
//
// State slot (6 words, in a DRAM state buffer):
//   [0] magic  [1] batch  [2] checksum(X)  [3] 0  [4] seed  [5] flags
// The X tile itself lives in the per-core region of a DRAM working buffer
// and is captured/restored by the runtime together with the slot.
//
// L1 layout at L1_UNRESERVED_BASE:
//   [ X tile ][ B tile ][ T tile ][ slot staging (6 words) ]

#include "dataflow_api.h"

#define SLOT_U32S 6
#define TILE_N    32
#define TILE_BYTES (TILE_N * TILE_N * 4)
#define MAGIC     0xC0DEC0DEu
#define FLAG_RESUME (1u << 1)

static uint32_t xorshift32(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

static uint32_t checksum_tile(const uint32_t *t) {
    uint32_t s = 0;
    for (uint32_t i = 0; i < TILE_N * TILE_N; i++) {
        s += t[i];
    }
    return s;
}

void kernel_main() {
    const uint32_t slot_dram_addr  = get_arg_val<uint32_t>(0);
    const uint32_t tile_dram_addr  = get_arg_val<uint32_t>(1);
    const uint32_t n_batches       = get_arg_val<uint32_t>(2);
    const uint32_t checkpoint_mode = get_arg_val<uint32_t>(3);
    const uint32_t rng_seed        = get_arg_val<uint32_t>(4);

    const uint32_t l1 = (uint32_t)L1_UNRESERVED_BASE;
    uint32_t *X = (uint32_t *)l1;
    uint32_t *B = (uint32_t *)(l1 + TILE_BYTES);
    uint32_t *T = (uint32_t *)(l1 + 2 * TILE_BYTES);
    volatile tt_l1_ptr uint32_t *slot =
        (volatile tt_l1_ptr uint32_t *)(l1 + 3 * TILE_BYTES);

    const uint64_t slot_noc = get_noc_addr(slot_dram_addr, 0);
    const uint64_t tile_noc = get_noc_addr(tile_dram_addr, 0);

    noc_async_read(tile_noc, l1, TILE_BYTES);                        // X
    noc_async_read(slot_noc, (uint32_t)slot, SLOT_U32S * 4);
    noc_async_read_barrier();

    uint32_t batch, seed;
    if (slot[5] & FLAG_RESUME) {
        batch = slot[1];
        seed  = slot[4];
        slot[5] = 0;
    } else {
        batch = 0;
        seed  = rng_seed | 1u;
    }

    // B is derived from the seed once per launch (deterministic).
    uint32_t bs = seed;
    for (uint32_t i = 0; i < TILE_N * TILE_N; i++) {
        B[i] = xorshift32(&bs);
    }

    const uint32_t target = checkpoint_mode ? (batch + 1) : n_batches;
    while (batch < target) {
        for (uint32_t r = 0; r < TILE_N; r++) {
            for (uint32_t c = 0; c < TILE_N; c++) {
                uint32_t acc = 0;
                for (uint32_t k = 0; k < TILE_N; k++) {
                    acc += X[r * TILE_N + k] * B[k * TILE_N + c];
                }
                T[r * TILE_N + c] = acc;
            }
        }
        // X = T
        for (uint32_t i = 0; i < TILE_N * TILE_N; i++) {
            X[i] = T[i];
        }
        batch++;
        if (checkpoint_mode) {
            noc_async_write(l1, tile_noc, TILE_BYTES);               // X out
            slot[1] = batch;
            slot[2] = checksum_tile(X);
            slot[3] = 0;
            slot[4] = seed;
            slot[5] = 0;
            noc_async_write((uint32_t)slot, slot_noc, SLOT_U32S * 4);
            noc_async_write_barrier();
        }
    }

    noc_async_write(l1, tile_noc, TILE_BYTES);                       // X out
    slot[0] = MAGIC;
    slot[1] = batch;
    slot[2] = checksum_tile(X);
    slot[3] = 0;
    slot[4] = seed;
    slot[5] = 0;
    noc_async_write((uint32_t)slot, slot_noc, SLOT_U32S * 4);
    noc_async_write_barrier();
}
