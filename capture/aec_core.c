#include "aec_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "speexdsp/speex/speex_echo.h"
#include "speexdsp/speex/speex_preprocess.h"

struct aec_core {
    SpeexEchoState *echo;
    SpeexPreprocessState *pre;
    int idle_frames;       // consecutive frames with a silent reference
    int idle_limit;        // frames after which no echo can remain
    int engaged;           // canceller ran on the previous frame
    float mic_gain;        // fixed pre-gain into the int16 canceller
    uint64_t processed, passed;
    spx_int16_t mic16[AEC_FRAME], ref16[AEC_FRAME], out16[AEC_FRAME];
};

// −72 dBFS: below this the speakers are not producing anything the mic can hear.
static const float kRefSilentRms = 2.5e-4f;

static inline spx_int16_t to16(float v) {
    float s = v * 32767.0f;
    if (s > 32767.0f) s = 32767.0f;
    if (s < -32768.0f) s = -32768.0f;
    return (spx_int16_t)lrintf(s);
}

aec_core *aec_new(int tail_samples, int echo_suppress_db, int echo_suppress_active_db, float mic_gain_db) {
    aec_core *a = calloc(1, sizeof *a);
    if (!a) return NULL;
    a->mic_gain = powf(10, mic_gain_db / 20);
    int rate = 16000;
    a->echo = speex_echo_state_init(AEC_FRAME, tail_samples);
    speex_echo_ctl(a->echo, SPEEX_ECHO_SET_SAMPLING_RATE, &rate);
    a->pre = speex_preprocess_state_init(AEC_FRAME, rate);
    int off = 0, on = 1;
    // Only residual-echo suppression: no denoise, no AGC, no VAD — the
    // daemon's thresholds and recognizers are calibrated on un-gained,
    // un-denoised mic audio.
    speex_preprocess_ctl(a->pre, SPEEX_PREPROCESS_SET_DENOISE, &off);
    speex_preprocess_ctl(a->pre, SPEEX_PREPROCESS_SET_AGC, &off);
    speex_preprocess_ctl(a->pre, SPEEX_PREPROCESS_SET_DEREVERB, &off);
    speex_preprocess_ctl(a->pre, SPEEX_PREPROCESS_SET_ECHO_STATE, a->echo);
    if (echo_suppress_db) speex_preprocess_ctl(a->pre, SPEEX_PREPROCESS_SET_ECHO_SUPPRESS, &echo_suppress_db);
    if (echo_suppress_active_db)
        speex_preprocess_ctl(a->pre, SPEEX_PREPROCESS_SET_ECHO_SUPPRESS_ACTIVE, &echo_suppress_active_db);
    (void)on;
    a->idle_limit = (tail_samples + 8000) / AEC_FRAME;  // tail + 500 ms
    a->idle_frames = a->idle_limit;                     // start idle
    return a;
}

void aec_free(aec_core *a) {
    if (!a) return;
    speex_echo_state_destroy(a->echo);
    speex_preprocess_state_destroy(a->pre);
    free(a);
}

int aec_process(aec_core *a, const float *mic, const float *ref, float *out) {
    float e = 0;
    for (int i = 0; i < AEC_FRAME; i++) e += ref[i] * ref[i];
    if (sqrtf(e / AEC_FRAME) >= kRefSilentRms) a->idle_frames = 0;
    else if (a->idle_frames < a->idle_limit) a->idle_frames++;
    if (a->idle_frames >= a->idle_limit) {
        for (int i = 0; i < AEC_FRAME; i++) out[i] = mic[i] * a->mic_gain;
        a->engaged = 0;
        a->passed++;
        return 0;
    }
    for (int i = 0; i < AEC_FRAME; i++) { a->mic16[i] = to16(mic[i] * a->mic_gain); a->ref16[i] = to16(ref[i]); }
    speex_echo_cancellation(a->echo, a->mic16, a->ref16, a->out16);
    speex_preprocess_run(a->pre, a->out16);
    for (int i = 0; i < AEC_FRAME; i++) out[i] = a->out16[i] * (1.0f / 32768.0f);
    a->engaged = 1;
    a->processed++;
    return 1;
}

void aec_stats(const aec_core *a, uint64_t *processed, uint64_t *passed) {
    if (processed) *processed = a->processed;
    if (passed) *passed = a->passed;
}

// ---- AGC --------------------------------------------------------------------
struct aec_agc {
    float target, max_gain_db, gain_db, noise, speech;  // linear rms levels
    float lim;                                          // limiter gain
};

aec_agc *aec_agc_new(float target_dbfs, float max_gain_db, float init_gain_db) {
    aec_agc *g = calloc(1, sizeof *g);
    g->target = powf(10, target_dbfs / 20);
    g->max_gain_db = max_gain_db;
    g->gain_db = init_gain_db;
    g->noise = 1e-4f;   // −80 dBFS until measured
    g->speech = 0;
    g->lim = 1;
    return g;
}

void aec_agc_free(aec_agc *g) { free(g); }

float aec_agc_gain_db(const aec_agc *g) { return g->gain_db; }

void aec_agc_process(aec_agc *g, float *x, int speech_ok) {
    float e = 0;
    for (int i = 0; i < AEC_FRAME; i++) e += x[i] * x[i];
    float rms = sqrtf(e / AEC_FRAME) + 1e-9f;
    // Noise floor: fast down, very slow up (~3 dB/s).
    g->noise = rms < g->noise ? 0.9f * g->noise + 0.1f * rms : g->noise * 1.0035f;
    // Speech frames: 12 dB above the floor and above −75 dBFS.
    if (speech_ok && rms > 4 * g->noise && rms > 1.8e-4f) {
        // Speech level: 2 s time constant over voiced frames only.
        g->speech = g->speech == 0 ? rms : 0.995f * g->speech + 0.005f * rms;
        float want = 20 * log10f(g->target / g->speech);
        if (want > g->max_gain_db) want = g->max_gain_db;
        if (want < 0) want = 0;
        float step = 0.06f;  // dB per voiced frame = 6 dB/s while he talks
        if (want > g->gain_db + step) g->gain_db += step;
        else if (want < g->gain_db - 4 * step) g->gain_db -= 4 * step;  // down 4x faster
        else g->gain_db = want;
    }
    float lin = powf(10, g->gain_db / 20);
    // Peak limiter at −1 dBFS: instant attack, ~100 ms release.
    float pk = 0;
    for (int i = 0; i < AEC_FRAME; i++) { float a = fabsf(x[i]) * lin; if (a > pk) pk = a; }
    float need = pk > 0.89f ? 0.89f / pk : 1.0f;
    g->lim = need < g->lim ? need : g->lim + (1 - g->lim) * 0.1f;
    if (g->lim > need) g->lim = need;
    lin *= g->lim;
    for (int i = 0; i < AEC_FRAME; i++) x[i] *= lin;
}
