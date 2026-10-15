# erle.py PREGAIN_DB DIR... : (aec_eval output <clip>.aecraw.wav rendered with that mic pre-gain, no AGC)
# ERLE (dB) where only echo is present: the 3 s media lead before he speaks
# (0.5-3 s, after initial convergence) and the 1 s tail after; plus near-end attenuation
# while he talks = level(out) vs level(near) on his voiced frames.
import sys, glob, numpy as np, scipy.io.wavfile as wf, warnings; warnings.filterwarnings('ignore')
def rd(p): return wf.read(p)[1].astype(np.float64) / 32768
PRE = float(sys.argv[1])
for d in sys.argv[2:]:
    e1, e2, att = [], [], []
    for m in sorted(glob.glob(d + '/*.mic.wav')):
        b = m[:-8]; x = rd(m); y = rd(b + '.aecraw.wav') / 10**(PRE/20); n = rd(b + '.near.wav')
        s, t = int(0.5*16000), 3*16000
        e1.append(10*np.log10(np.mean(x[s:t]**2) / np.mean(y[s:t]**2)))
        e2.append(10*np.log10(np.mean(x[-16000:]**2) / np.mean(y[-16000:]**2)))
    print(f"{d.split('/')[-1]:12s} ERLE lead(0.5-3s) {np.mean(e1):5.1f} dB  ERLE end-tail(after speech) {np.mean(e2):5.1f} dB  n={len(e1)}")
