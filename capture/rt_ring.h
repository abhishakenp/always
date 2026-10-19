// Lock-free single-producer / single-consumer ring of mic blocks, written
// from the CoreAudio IO thread (AVAudioSinkNode) and read by the helper's
// processing thread. The producer never blocks, allocates or locks.
#pragma once
#include <stdint.h>
#include <stddef.h>

typedef struct rt_ring rt_ring;

// `capacity_floats` is rounded up to a power of two.
rt_ring *rt_ring_create(size_t capacity_floats);
// Producer: one block (n samples of channel 0, `stride` floats apart:
// 1 for a deinterleaved buffer, the channel count for interleaved) captured at `host`. Returns 0
// when it did not fit (the block is dropped and counted).
int rt_ring_write(rt_ring *r, const float *p, uint32_t n, uint32_t stride, uint64_t host);
// Consumer: the next block into `out` (room for `max` samples); its length,
// 0 when empty. `*host` gets its capture time.
uint32_t rt_ring_read(rt_ring *r, float *out, uint32_t max, uint64_t *host);
// Blocks dropped because the ring was full (since the last call; resets).
uint64_t rt_ring_take_dropped(rt_ring *r);
