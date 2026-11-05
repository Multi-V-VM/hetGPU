// SPDX-License-Identifier: Apache-2.0
//
// Iterative vector-add-accumulate worker kernel for the TT->TT
// live-migration experiment. Per batch: C += A, where A is a constant tile
// derived from the kernel seed and streamed from its own DRAM region.
// Bandwidth-bound contrast to the matmul kernel.
//
// State slot (6 words, in a DRAM state buffer):
//   [0] magic  [1] batch  [2] checksum(C)  [3] 0  [4] seed  [5] flags
// Working buffer per core: [ C tile (TILE_BYTES) ][ A tile (TILE_BYTES) ].
// Both tiles are captured/restored by the runtime with the slot.

#include "dataflow_api.h"

#define SLOT_U32S 6
#define TILE_ELEMS 2048
#define TILE_BYTES (TILE_ELEMS * 4)
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
    for (uint32_t i = 0; i < TILE_ELEMS; i++) {
        s += t[i];
    }
    return s;
}

void kernel_main() {
    const uint32_t slot_dram_addr  = get_arg_val<uint32_t>(0);
    const uint32_t tile_dram_addr  = get_arg_val<uint32_t>(1);  // C region
    const uint32_t n_batches       = get_arg_val<uint32_t>(2);
    const uint32_t checkpoint_mode = get_arg_val<uint32_t>(3);
    const uint32_t rng_seed        = get_arg_val<uint32_t>(4);

    const uint32_t l1 = (uint32_t)L1_UNRESERVED_BASE;
    uint32_t *C = (uint32_t *)l1;
    uint32_t *A = (uint32_t *)(l1 + TILE_BYTES);
    volatile tt_l1_ptr uint32_t *slot =
        (volatile tt_l1_ptr uint32_t *)(l1 + 2 * TILE_BYTES);

    const uint64_t slot_noc = get_noc_addr(slot_dram_addr, 0);
    const uint64_t c_noc = get_noc_addr(tile_dram_addr, 0);
    const uint64_t a_noc = get_noc_addr(tile_dram_addr + TILE_BYTES, 0);

    noc_async_read(c_noc, l1, TILE_BYTES);
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

    // A is a deterministic function of the seed, so the kernel restores it
    // itself on resume without needing it in the serialized state.
    uint32_t as = seed ^ 0xA5A5A5A5u;
    for (uint32_t i = 0; i < TILE_ELEMS; i++) {
        A[i] = xorshift32(&as);
    }

    const uint32_t target = checkpoint_mode ? (batch + 1) : n_batches;
    while (batch < target) {
        for (uint32_t i = 0; i < TILE_ELEMS; i++) {
            C[i] += A[i];
        }
        batch++;
        if (checkpoint_mode) {
            noc_async_write(l1, c_noc, TILE_BYTES);                  // C out
            slot[1] = batch;
            slot[2] = checksum_tile(C);
            slot[3] = 0;
            slot[4] = seed;
            slot[5] = 0;
            noc_async_write((uint32_t)slot, slot_noc, SLOT_U32S * 4);
            noc_async_write_barrier();
        }
    }

    noc_async_write(l1, c_noc, TILE_BYTES);
    slot[0] = MAGIC;
    slot[1] = batch;
    slot[2] = checksum_tile(C);
    slot[3] = 0;
    slot[4] = seed;
    slot[5] = 0;
    noc_async_write((uint32_t)slot, slot_noc, SLOT_U32S * 4);
    noc_async_write_barrier();
}
