// tse_core.c — see tse_core.h.
//
// Weights file ("TSE1"): 4-byte magic, int32 F, H, D, then float32 arrays in
// PyTorch layout (row-major [out][in]):
//   inp.weight [H][2F], inp.bias [H], film.weight [2H][D], film.bias [2H],
//   g1.weight_ih [3H][H], g1.weight_hh [3H][H], g1.bias_ih [3H], g1.bias_hh [3H],
//   g2.(same), out.weight [F][H], out.bias [F]
// GRU gate order (PyTorch): r, z, n.
#include "tse_core.h"

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "speexdsp/smallft.h"

#define N 320
#define HOPN 160

struct tse_core {
    int F, H, D;
    float *inp_w, *inp_b, *film_w, *film_b;
    float *g_wih[2], *g_whh[2], *g_bih[2], *g_bhh[2];
    float *out_w, *out_b;
    float *gam, *bet;           // FiLM from the voiceprint (precomputed)
    float *h1, *h2, *x, *gi, *gh, *y, *mask, *feat;
    float win[N];
    float ebuf[N], rbuf[N];     // last 320 input samples
    float ola[N];               // overlap-add accumulator
    float spec[N], tmp[N];
    struct drft_lookup fft;
    float floor;
    float *blob;
};

static float sigm(float v) { return 1.f / (1.f + expf(-v)); }

void tse_reset(tse_core *t) {
    memset(t->h1, 0, sizeof(float) * t->H);
    memset(t->h2, 0, sizeof(float) * t->H);
    memset(t->ebuf, 0, sizeof t->ebuf);
    memset(t->rbuf, 0, sizeof t->rbuf);
    memset(t->ola, 0, sizeof t->ola);
}

void tse_set_floor(tse_core *t, float floor) { t->floor = floor; }

tse_core *tse_new(const char *path, const float *vp, int dim) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[4];
    int hdr[3];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TSE1", 4) || fread(hdr, 4, 3, f) != 3) { fclose(f); return NULL; }
    int F = hdr[0], H = hdr[1], D = hdr[2];
    if (F != N / 2 + 1 || H <= 0 || H > 1024 || D != dim) { fclose(f); return NULL; }
    size_t n = (size_t)H * 2 * F + H + 2 * H * D + 2 * H + 2 * (2 * 3 * H * H + 2 * 3 * H) + F * H + F;
    float *blob = malloc(n * sizeof(float));
    if (!blob || fread(blob, sizeof(float), n, f) != n) { free(blob); fclose(f); return NULL; }
    char extra;
    if (fread(&extra, 1, 1, f) != 0) { free(blob); fclose(f); return NULL; }  // trailing bytes: wrong file
    fclose(f);
    tse_core *t = calloc(1, sizeof *t);
    t->F = F; t->H = H; t->D = D; t->blob = blob; t->floor = 0.03f;
    float *p = blob;
    t->inp_w = p; p += H * 2 * F; t->inp_b = p; p += H;
    t->film_w = p; p += 2 * H * D; t->film_b = p; p += 2 * H;
    for (int g = 0; g < 2; g++) {
        t->g_wih[g] = p; p += 3 * H * H; t->g_whh[g] = p; p += 3 * H * H;
        t->g_bih[g] = p; p += 3 * H; t->g_bhh[g] = p; p += 3 * H;
    }
    t->out_w = p; p += F * H; t->out_b = p; p += F;
    t->gam = calloc(2 * H, sizeof(float)); t->bet = t->gam + H;
    float *v = malloc(sizeof(float) * D), norm = 0;
    for (int i = 0; i < D; i++) norm += vp[i] * vp[i];
    norm = sqrtf(norm) + 1e-9f;
    for (int i = 0; i < D; i++) v[i] = vp[i] / norm;
    memcpy(t->gam, t->film_b, sizeof(float) * 2 * H);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 2 * H, D, 1.f, t->film_w, D, v, 1, 1.f, t->gam, 1);
    free(v);
    t->h1 = calloc(H, sizeof(float)); t->h2 = calloc(H, sizeof(float)); t->x = calloc(H, sizeof(float));
    t->gi = calloc(3 * H, sizeof(float)); t->gh = calloc(3 * H, sizeof(float)); t->y = calloc(H, sizeof(float));
    t->mask = calloc(F, sizeof(float)); t->feat = calloc(2 * F, sizeof(float));
    for (int i = 0; i < N; i++) t->win[i] = sqrtf(0.5f - 0.5f * cosf(2 * (float)M_PI * i / N));  // periodic sqrt-Hann
    spx_drft_init(&t->fft, N);
    return t;
}

void tse_free(tse_core *t) {
    if (!t) return;
    spx_drft_clear(&t->fft);
    free(t->blob); free(t->gam); free(t->h1); free(t->h2); free(t->x); free(t->gi); free(t->gh);
    free(t->y); free(t->mask); free(t->feat);
    free(t);
}

// half-complex (smallft) power -> log10 feature, matching model.feat()
static void logpow(const float *hc, float *out) {
    out[0] = (log10f(hc[0] * hc[0] + 1e-9f) + 5) / 2.5f;
    for (int k = 1; k < N / 2; k++) {
        float re = hc[2 * k - 1], im = hc[2 * k];
        out[k] = (log10f(re * re + im * im + 1e-9f) + 5) / 2.5f;
    }
    out[N / 2] = (log10f(hc[N - 1] * hc[N - 1] + 1e-9f) + 5) / 2.5f;
}

static void gru(tse_core *t, int g, const float *x, float *h) {
    int H = t->H;
    memcpy(t->gi, t->g_bih[g], sizeof(float) * 3 * H);
    memcpy(t->gh, t->g_bhh[g], sizeof(float) * 3 * H);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 3 * H, H, 1.f, t->g_wih[g], H, x, 1, 1.f, t->gi, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 3 * H, H, 1.f, t->g_whh[g], H, h, 1, 1.f, t->gh, 1);
    for (int i = 0; i < H; i++) {
        float r = sigm(t->gi[i] + t->gh[i]);
        float z = sigm(t->gi[H + i] + t->gh[H + i]);
        float n = tanhf(t->gi[2 * H + i] + r * t->gh[2 * H + i]);
        h[i] = (1 - z) * n + z * h[i];
    }
}

void tse_process(tse_core *t, const float *e, const float *r, float *out) {
    int F = t->F, H = t->H;
    memmove(t->ebuf, t->ebuf + HOPN, sizeof(float) * (N - HOPN)); memcpy(t->ebuf + N - HOPN, e, sizeof(float) * HOPN);
    memmove(t->rbuf, t->rbuf + HOPN, sizeof(float) * (N - HOPN)); memcpy(t->rbuf + N - HOPN, r, sizeof(float) * HOPN);
    // reference spectrum -> features[F..2F)
    for (int i = 0; i < N; i++) t->tmp[i] = t->rbuf[i] * t->win[i];
    spx_drft_forward(&t->fft, t->tmp);
    logpow(t->tmp, t->feat + F);
    // canceller-output spectrum (kept for masking) -> features[0..F)
    for (int i = 0; i < N; i++) t->spec[i] = t->ebuf[i] * t->win[i];
    spx_drft_forward(&t->fft, t->spec);
    logpow(t->spec, t->feat);
    // input layer + FiLM + relu
    memcpy(t->x, t->inp_b, sizeof(float) * H);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, H, 2 * F, 1.f, t->inp_w, 2 * F, t->feat, 1, 1.f, t->x, 1);
    for (int i = 0; i < H; i++) { float v = t->x[i] * (1 + t->gam[i]) + t->bet[i]; t->x[i] = v > 0 ? v : 0; }
    gru(t, 0, t->x, t->h1);
    gru(t, 1, t->h1, t->h2);
    for (int i = 0; i < H; i++) t->y[i] = t->h1[i] + t->h2[i];
    memcpy(t->mask, t->out_b, sizeof(float) * F);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, F, H, 1.f, t->out_w, H, t->y, 1, 1.f, t->mask, 1);
    for (int k = 0; k < F; k++) { float m = sigm(t->mask[k]); t->mask[k] = m < t->floor ? t->floor : m; }
    // apply, inverse, overlap-add
    t->spec[0] *= t->mask[0];
    for (int k = 1; k < N / 2; k++) { t->spec[2 * k - 1] *= t->mask[k]; t->spec[2 * k] *= t->mask[k]; }
    t->spec[N - 1] *= t->mask[N / 2];
    spx_drft_backward(&t->fft, t->spec);
    for (int i = 0; i < N; i++) t->ola[i] += t->spec[i] * (1.f / N) * t->win[i];
    memcpy(out, t->ola, sizeof(float) * HOPN);
    memmove(t->ola, t->ola + HOPN, sizeof(float) * (N - HOPN));
    memset(t->ola + N - HOPN, 0, sizeof(float) * HOPN);
}
