#include "rt_ring.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

// Each block: [n (as float bits)][host lo][host hi][n samples], in floats.
#define HDR 3

struct rt_ring {
    float *buf;
    size_t mask;
    _Atomic size_t head;  // next write (producer)
    _Atomic size_t tail;  // next read (consumer)
    _Atomic uint64_t dropped;
};

rt_ring *rt_ring_create(size_t cap) {
    size_t c = 1;
    while (c < cap) c <<= 1;
    rt_ring *r = calloc(1, sizeof *r);
    if (!r) return NULL;
    r->buf = calloc(c, sizeof(float));
    if (!r->buf) { free(r); return NULL; }
    r->mask = c - 1;
    atomic_init(&r->head, 0);
    atomic_init(&r->tail, 0);
    atomic_init(&r->dropped, 0);
    return r;
}

static void put(rt_ring *r, size_t at, const float *p, size_t n) {
    for (size_t i = 0; i < n; i++) r->buf[(at + i) & r->mask] = p[i];
}

static void get(const rt_ring *r, size_t at, float *p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = r->buf[(at + i) & r->mask];
}

int rt_ring_write(rt_ring *r, const float *p, uint32_t n, uint32_t stride, uint64_t host) {
    size_t head = atomic_load_explicit(&r->head, memory_order_relaxed);
    size_t tail = atomic_load_explicit(&r->tail, memory_order_acquire);
    size_t need = HDR + (size_t)n;
    if ((r->mask + 1) - (head - tail) < need) {
        atomic_fetch_add_explicit(&r->dropped, 1, memory_order_relaxed);
        return 0;
    }
    float hdr[HDR];
    uint32_t lo = (uint32_t)host, hi = (uint32_t)(host >> 32);
    memcpy(&hdr[0], &n, 4);
    memcpy(&hdr[1], &lo, 4);
    memcpy(&hdr[2], &hi, 4);
    put(r, head, hdr, HDR);
    if (stride <= 1) put(r, head + HDR, p, n);
    else for (uint32_t i = 0; i < n; i++) r->buf[(head + HDR + i) & r->mask] = p[(size_t)i * stride];
    atomic_store_explicit(&r->head, head + need, memory_order_release);
    return 1;
}

uint32_t rt_ring_read(rt_ring *r, float *out, uint32_t max, uint64_t *host) {
    size_t tail = atomic_load_explicit(&r->tail, memory_order_relaxed);
    size_t head = atomic_load_explicit(&r->head, memory_order_acquire);
    if (head == tail) return 0;
    float hdr[HDR];
    get(r, tail, hdr, HDR);
    uint32_t n, lo, hi;
    memcpy(&n, &hdr[0], 4);
    memcpy(&lo, &hdr[1], 4);
    memcpy(&hi, &hdr[2], 4);
    *host = ((uint64_t)hi << 32) | lo;
    uint32_t take = n < max ? n : max;  // a block larger than `max` is truncated
    get(r, tail + HDR, out, take);
    atomic_store_explicit(&r->tail, tail + HDR + n, memory_order_release);
    return take;
}

uint64_t rt_ring_take_dropped(rt_ring *r) {
    return atomic_exchange_explicit(&r->dropped, 0, memory_order_relaxed);
}
