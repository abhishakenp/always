// Offline harness: a synthetic shout through the capture AGC + limiter. Never touches a mic.
// Signal: voiced speech-like source (harmonics of 220 Hz, 4 Hz syllable AM) at a quiet level for
// QUIET_S seconds so the AGC sits at its start gain, then a sudden shout 40 dB louder whose
// post-pregain peak is -2 dBFS (the logged raw mic peak +10.5 dBFS, pregain -12 dB).
// Prints, over the shout: limiter pumping (std of the per-frame applied gain, dB), the largest
// frame-to-frame gain jump, THD+N (residual after removing the best-fit scaled copy of the input,
// so only time-varying gain / clipping shows), and the output peak. Frames are 10 ms, 16 kHz.
// `speech_ok`=0 reproduces tap mode with media playing (canceller running: adaptation frozen).
// Build: see agc_shout_eval.sh. Usage: agc_shout_eval [speech_ok 0|1]
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aec_core.h"

#define FR 160
#define SR 16000
static float synth(int n, float amp) {
    float t = (float)n / SR, s = 0;
    for (int k = 1; k <= 8; k++) s += sinf(6.2831853f * 220.0f * k * t) / k;
    float env = 0.65f + 0.35f * sinf(6.2831853f * 4.0f * t);
    return amp * s * env;
}
int main(int argc, char **argv) {
    int speech_ok = argc > 1 ? atoi(argv[1]) : 0;
    float pre = -12;
    aec_agc *g = aec_agc_new(-26, 35 - pre, 22 - pre);  // exactly the capture helper's arguments
    int quiet_frames = 600, shout_frames = 300, N = quiet_frames + shout_frames;
    // Normalise so the shout's peak is -2 dBFS, the quiet part 40 dB below.
    float pk = 0;
    for (int i = 0; i < shout_frames * FR; i++) pk = fmaxf(pk, fabsf(synth(i, 1)));
    float a_shout = powf(10, -2.0f / 20) / pk, a_quiet = a_shout * powf(10, -40.0f / 20);
    float *x = calloc((size_t)N * FR, 4), *y = calloc((size_t)N * FR, 4);
    float *gdb = calloc(N, 4);
    for (int f = 0; f < N; f++) {
        float a = f < quiet_frames ? a_quiet : a_shout;
        for (int i = 0; i < FR; i++) x[f * FR + i] = synth(f * FR + i, a);
        memcpy(&y[f * FR], &x[f * FR], FR * 4);
        aec_agc_process(g, &y[f * FR], speech_ok);
        double ei = 0, eo = 0;
        for (int i = 0; i < FR; i++) { ei += x[f*FR+i]*x[f*FR+i]; eo += y[f*FR+i]*y[f*FR+i]; }
        gdb[f] = ei > 1e-12 ? 10 * log10(eo / ei) : 0;
    }
    int s0 = quiet_frames, s1 = N;
    double mean = 0, var = 0, jump = 0;
    for (int f = s0; f < s1; f++) mean += gdb[f];
    mean /= (s1 - s0);
    for (int f = s0; f < s1; f++) { var += (gdb[f] - mean) * (gdb[f] - mean); if (f > s0) jump = fmax(jump, fabs(gdb[f] - gdb[f-1])); }
    double sxy = 0, sxx = 0, peak = 0;
    for (int i = s0 * FR; i < s1 * FR; i++) { sxy += (double)x[i] * y[i]; sxx += (double)x[i] * x[i]; peak = fmax(peak, fabs(y[i])); }
    double a = sxy / sxx, res = 0;
    for (int i = s0 * FR; i < s1 * FR; i++) { double d = y[i] - a * x[i]; res += d * d; }
    int clip = 0;
    for (int i = s0 * FR; i < s1 * FR; i++) if (fabsf(y[i]) >= 0.999f) clip++;
    printf("speech_ok=%d  shout gain mean %.1f dB  pumping (std of per-frame gain) %.2f dB  max frame-to-frame jump %.1f dB\n",
           speech_ok, mean, sqrt(var / (s1 - s0)), jump);
    printf("            THD+N %.1f dB  output peak %.2f dBFS  clipped samples %d  final AGC gain %.1f dB\n",
           10 * log10(res / (a * a * sxx)), 20 * log10(peak), clip, aec_agc_gain_db(g));
    return 0;
}
