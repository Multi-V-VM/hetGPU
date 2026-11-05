// SPDX-License-Identifier: Apache-2.0
//
// Monte Carlo pi worker kernel for the TT->TT live-migration experiment on
// simulated Blackhole (ttsim). One worker core per kernel instance.
//
// Each core owns one 6-word state slot in a DRAM buffer:
//   [0] magic   [1] batch   [2] hits   [3] samples   [4] rng   [5] flags
// flags bit0 = pause-after-batch request (written by the runtime)
// flags bit1 = resume request (written by the runtime)
//
// checkpoint_mode == 1 runs the cooperative protocol: restore-or-init,
// process ONE batch, publish state to DRAM, exit (the runtime captures the
// slot and relaunches with a resume state). This is the instrumented /
// migration configuration.
// checkpoint_mode == 0 runs all batches in one launch and only publishes
// the final state: the baseline configuration, no checkpoint instrumentation.

#include "dataflow_api.h"

#define SLOT_U32S 6
#define MAGIC     0xC0DEC0DEu
#define FLAG_PAUSE  (1u << 0)
#define FLAG_RESUME (1u << 1)

static uint32_t xorshift32(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

void kernel_main() {
    // Runtime args: 0 slot dram address, 1 samples per batch, 2 total batches,
    //               3 checkpoint_mode, 4 rng seed
    const uint32_t slot_dram_addr  = get_arg_val<uint32_t>(0);
    const uint32_t batch_samples   = get_arg_val<uint32_t>(1);
    const uint32_t n_batches       = get_arg_val<uint32_t>(2);
    const uint32_t checkpoint_mode = get_arg_val<uint32_t>(3);
    const uint32_t rng_seed        = get_arg_val<uint32_t>(4);

    uint32_t l1_scratch = (uint32_t)L1_UNRESERVED_BASE;
    volatile tt_l1_ptr uint32_t *slot = (volatile tt_l1_ptr uint32_t *)l1_scratch;

    // Pull the current slot; the host may have planted a resume state here.
    const uint64_t slot_noc = get_noc_addr(slot_dram_addr, 0);
    noc_async_read(slot_noc, l1_scratch, SLOT_U32S * sizeof(uint32_t));
    noc_async_read_barrier();

    uint32_t batch, hits, samples, rng;
    if (slot[5] & FLAG_RESUME) {
        batch   = slot[1];
        hits    = slot[2];
        samples = slot[3];
        rng     = slot[4];
        slot[5] = 0;  // consume the resume flag
    } else {
        batch = 0;
        hits = 0;
        samples = 0;
        rng = rng_seed | 1u;
    }

    const uint32_t target = checkpoint_mode ? (batch + 1) : n_batches;
    while (batch < target) {
        uint32_t batch_hits = 0;
        for (uint32_t i = 0; i < batch_samples; i++) {
            uint32_t a = xorshift32(&rng);
            uint32_t b = xorshift32(&rng);
            float u = (float)a * (1.0f / 4294967296.0f);
            float v = (float)b * (1.0f / 4294967296.0f);
            if (u * u + v * v < 1.0f) {
                batch_hits++;
            }
        }
        hits += batch_hits;
        samples += batch_samples;
        batch++;
        if (checkpoint_mode) {
            slot[1] = batch;
            slot[2] = hits;
            slot[3] = samples;
            slot[4] = rng;
            noc_async_write(l1_scratch, slot_noc, SLOT_U32S * sizeof(uint32_t));
            noc_async_write_barrier();
        }
    }

    slot[0] = MAGIC;
    slot[1] = batch;
    slot[2] = hits;
    slot[3] = samples;
    slot[4] = rng;
    slot[5] = 0;
    noc_async_write(l1_scratch, slot_noc, SLOT_U32S * sizeof(uint32_t));
    noc_async_write_barrier();
}
