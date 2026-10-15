// aec_core — reference-based acoustic echo cancellation for always-capture.
//
// The far-end reference is what the Mac itself is playing (a CoreAudio
// process tap of all output), so ANY process's audio — a YouTube video, music,
// Iris's speech, Always's own cues — is subtracted from the microphone.
// SpeexDSP MDF adaptive filter + residual-echo suppression, 16 kHz mono,
// 10 ms frames. Shared by always-capture and the offline eval (aec_eval.c).
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aec_core aec_core;

#define AEC_FRAME 160  // 10 ms at 16 kHz

/// tail_samples: echo path length the filter models (e.g. 4096 = 256 ms).
/// echo_suppress_db / active_db: residual echo suppression (negative dB;
/// speex defaults are -40 / -15). 0 = speex default.
/// mic_gain_db: fixed gain on the mic before cancelling (and on pass-through);
/// 0 in always-capture (measured offline: 0 vs 18 dB pre-gain, same ERLE).
aec_core *aec_new(int tail_samples, int echo_suppress_db, int echo_suppress_active_db, float mic_gain_db);
void aec_free(aec_core *a);

/// One 10 ms frame: mic and ref are AEC_FRAME floats in [-1, 1]; out receives
/// AEC_FRAME floats. When the reference has been silent long enough that no
/// echo can remain (tail + 500 ms), the frame is passed through untouched and
/// costs nothing — the adaptive filter keeps its converged state.
/// Returns 1 if the canceller ran on this frame, 0 if passed through.
int aec_process(aec_core *a, const float *mic, const float *ref, float *out);

/// Frames processed / passed through since creation.
void aec_stats(const aec_core *a, uint64_t *processed, uint64_t *passed);

#ifdef __cplusplus
}
#endif

// ---- Speech-level AGC ------------------------------------------------------
// The raw built-in mic delivers his speech ~25 dB below what the voice-
// processing unit used to hand the daemon (measured: raw −53…−60 dBFS rms vs
// processed −20…−35), and far-field speech another 10–20 dB lower — too
// quiet for the recognizers. One slow gain brings SPEECH (not the noise
// floor) to `target_dbfs`: it adapts only on frames well above the tracked
// noise floor, moves at most `rate_db_s`, is capped at `max_gain_db`, and a
// peak limiter keeps it from clipping. 10 ms frames, 16 kHz.
typedef struct aec_agc aec_agc;
#ifdef __cplusplus
extern "C" {
#endif
aec_agc *aec_agc_new(float target_dbfs, float max_gain_db, float init_gain_db);
void aec_agc_free(aec_agc *g);
/// In place. `speech_ok` = 0 freezes adaptation (e.g. while the canceller is
/// still removing loud far-end audio, so music never trains the gain).
void aec_agc_process(aec_agc *g, float *frame, int speech_ok);
float aec_agc_gain_db(const aec_agc *g);
#ifdef __cplusplus
}
#endif
