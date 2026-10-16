// tse_core — his personal target-speaker extractor, run on the canceller output.
//
// Causal and streaming: 20 ms sqrt-Hann window, 10 ms hop at 16 kHz (161 bins).
// Per frame the features are log|E|^2 (canceller output) and log|R|^2 (the
// playback reference the canceller got). A FiLM layer conditioned on his
// enrolled voiceprint (voiceprint.json, 256-d WeSpeaker) feeds two GRU(128)
// layers, and their summed output gives a 161-bin mask applied to E. The
// weights are trained offline (~/proj/wt/far-room). Latency is one 10 ms hop
// on top of the 10 ms frame. Only run while the reference plays (the
// canceller is engaged); a quiet room never reaches it.
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
typedef struct tse_core tse_core;
/// Loads `weights_path` (format "TSE1", see tse_core.c) and conditions on
/// `voiceprint` (256 floats; L2-normalised here). NULL on any error.
tse_core *tse_new(const char *weights_path, const float *voiceprint, int dim);
void tse_free(tse_core *t);
/// One 10 ms frame (160 samples): e = canceller output, r = reference at the
/// same time. Writes 160 samples to out, delayed by one frame.
void tse_process(tse_core *t, const float *e, const float *r, float *out);
/// Clears the streaming state (GRU, overlap buffers): call when the
/// canceller disengages so the next engagement starts clean.
void tse_reset(tse_core *t);
/// Lower bound on the mask (linear, default 0.03 = -30 dB).
void tse_set_floor(tse_core *t, float floor);
#ifdef __cplusplus
}
#endif
