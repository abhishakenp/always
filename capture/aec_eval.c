// aec_eval — run aec_core offline: aec_eval MIC.wav REF.wav OUT.wav [tail] [supp] [supp_active]
// 16 kHz mono s16 WAVs. Prints processing cost (CPU µs per second of audio).
// Build: see capture/build.sh (aec_eval target).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "aec_core.h"

static int16_t *read_wav(const char *path, long *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    char id[4]; uint32_t sz; fread(id, 1, 4, f); fread(&sz, 4, 1, f); fread(id, 1, 4, f);
    for (;;) {
        if (fread(id, 1, 4, f) != 4 || fread(&sz, 4, 1, f) != 1) { fprintf(stderr, "%s: no data\n", path); exit(2); }
        if (!memcmp(id, "data", 4)) break;
        fseek(f, sz, SEEK_CUR);
    }
    int16_t *d = malloc(sz);
    *n = (long)fread(d, 2, sz / 2, f);
    fclose(f);
    return d;
}

static void write_wav(const char *path, const int16_t *d, long n) {
    FILE *f = fopen(path, "wb");
    uint32_t data = (uint32_t)(n * 2), riff = 36 + data, fmt = 16, rate = 16000, br = 32000;
    uint16_t pcm = 1, ch = 1, ba = 2, bits = 16;
    fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVEfmt ", 1, 8, f); fwrite(&fmt, 4, 1, f);
    fwrite(&pcm, 2, 1, f); fwrite(&ch, 2, 1, f); fwrite(&rate, 4, 1, f); fwrite(&br, 4, 1, f);
    fwrite(&ba, 2, 1, f); fwrite(&bits, 2, 1, f); fwrite("data", 1, 4, f); fwrite(&data, 4, 1, f);
    fwrite(d, 2, n, f);
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: aec_eval MIC REF OUT [tail] [supp] [supp_active] [agc_init_db|-] [noaec] [mic_pregain_db]\n"); return 1; }
    long nm, nr;
    int16_t *mic = read_wav(argv[1], &nm), *ref = read_wav(argv[2], &nr);
    int tail = argc > 4 ? atoi(argv[4]) : 4096;
    int supp = argc > 5 ? atoi(argv[5]) : 0, act = argc > 6 ? atoi(argv[6]) : 0;
    float pre = argc > 9 ? atof(argv[9]) : 0;
    aec_core *a = aec_new(tail, supp, act, pre);
    // argv[7]: AGC initial gain dB ("-" = no AGC); argv[8]: 1 = bypass AEC (AGC only)
    aec_agc *g = (argc > 7 && argv[7][0] != '-') ? aec_agc_new(-26, 35 - pre, atof(argv[7])) : NULL;
    int noaec = argc > 8 && atoi(argv[8]);
    int16_t *out = calloc(nm, 2);
    float m[AEC_FRAME], r[AEC_FRAME], o[AEC_FRAME];
    clock_t t0 = clock();
    for (long i = 0; i + AEC_FRAME <= nm; i += AEC_FRAME) {
        for (int k = 0; k < AEC_FRAME; k++) {
            m[k] = mic[i + k] / 32768.0f;
            r[k] = (i + k < nr) ? ref[i + k] / 32768.0f : 0;
        }
        int ran = 0;
        if (noaec) memcpy(o, m, sizeof o); else ran = aec_process(a, m, r, o);
        // As in always-capture: the gain adapts only while nothing plays.
        if (g) aec_agc_process(g, o, noaec ? 1 : !ran);
        for (int k = 0; k < AEC_FRAME; k++) {
            float s = o[k] * 32768.0f;
            out[i + k] = (int16_t)(s > 32767 ? 32767 : s < -32768 ? -32768 : s);
        }
    }
    double cpu = (double)(clock() - t0) / CLOCKS_PER_SEC;
    uint64_t pr, pa;
    aec_stats(a, &pr, &pa);
    write_wav(argv[3], out, nm);
    printf("audio_s=%.2f cpu_ms=%.1f cpu_us_per_audio_s=%.0f frames_processed=%llu passed=%llu\n",
           nm / 16000.0, cpu * 1000, cpu * 1e6 / (nm / 16000.0), pr, pa);
    aec_free(a);
    return 0;
}
