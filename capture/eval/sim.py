# sim.py OUTDIR CLIP... : build mic/ref pairs per condition (16 kHz mono).
# mic = near-end speech (his clip, optionally far-field/slow) + echo + noise floor
# echo = ref -> laptop-speaker soft clip -> room IR -> +30 ms system delay
# ref = what the Mac plays (what a process tap delivers)
import sys, os, numpy as np, scipy.io.wavfile as wf, scipy.signal as ss, subprocess
rng = np.random.default_rng(7)
SR = 16000
DRIVE = float(os.environ.get("DRIVE", "1.0"))
import warnings; warnings.filterwarnings("ignore")
out = sys.argv[1]; clips = sys.argv[2:]
src = os.path.join(os.path.dirname(__file__), 'src')
def rd(p):
    r, x = wf.read(p); assert r == SR; return x.astype(np.float32) / 32768
def wr(p, x): wf.write(p, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
def active_rms(x):
    f = x[:len(x)//160*160].reshape(-1, 160); e = np.sqrt((f**2).mean(1)); a = e[e > e.max() * 0.05]
    return float(np.sqrt((a**2).mean())) if len(a) else 1e-9
def db(x): return 10 ** (x / 20)
def ir(rt60, n, direct=1.0, seed=0):
    g = np.random.default_rng(seed); t = np.arange(n) / SR
    h = g.standard_normal(n) * np.exp(-6.91 * t / rt60) * 0.25
    h[0] += direct; return (h / np.sqrt((h**2).sum())).astype(np.float32)
SPK = ir(0.25, 4000, direct=1.5, seed=1)     # speaker -> mic path (laptop, small room)
ROOM = ir(0.5, 8000, direct=0.6, seed=2)     # far-field talker -> mic (reverberant)
def echo(ref):
    y = ref.copy() if DRIVE == 0 else np.tanh(DRIVE * ref / (np.abs(ref).max() + 1e-9)) * (np.abs(ref).max())  # speaker nonlinearity
    y = ss.fftconvolve(y, SPK)[:len(ref)]
    return np.concatenate([np.zeros(480, np.float32), y[:-480]])         # +30 ms
NEAR_DB = float(os.environ.get("NEAR_DB", "-28"))
conds = {  # name: (media, SER dB, far_field, slow)
    'clean': (None, 0, False, False),
    'music': ('music1', 0, False, False),
    'music_loud': ('music2', -6, False, False),
    'video': ('narr', -3, False, False),
    'iris': ('iris', -8, False, False),
    'far_music': ('music1', -12, True, False),
    'far_iris': ('iris', -14, True, False),
    'slow_video': ('narr', -3, False, True),
}
# Warm-up pair: 30 s of a DIFFERENT track through the same echo path. In real
# use the canceller's filter persists across playback (same speakers, same mic),
# so it is already converged when he talks over a new video or Iris; aec_eval
# processes this pair first (output discarded) when given WARM_PREFIX.
wm = rd(os.path.join(src, 'music2.wav'))[SR*60:SR*90]
we = echo(wm); we = we * db(NEAR_DB) / active_rms(we)
wr(os.path.join(out, 'warm.mic.wav'), we + rng.standard_normal(len(we)).astype(np.float32) * db(-70))
wr(os.path.join(out, 'warm.ref.wav'), wm * db(-20) / active_rms(wm))
for c in clips:
    base = os.path.splitext(os.path.basename(c))[0]
    near0 = rd(c)
    slow_p = os.path.join(out, base + '.slow.wav')
    if not os.path.exists(slow_p):
        subprocess.run(['sox', c, slow_p, 'tempo', '-s', '0.7'], check=True)
    for name, (media, ser, far, slow) in conds.items():
        near = rd(slow_p) if slow else near0.copy()
        if far: near = ss.fftconvolve(near, ROOM)[:len(near)]
        near = near * db(NEAR_DB - (12 if far else 0)) / active_rms(near)
        n = len(near) + SR  # 1 s tail
        near = np.concatenate([near, np.zeros(SR, np.float32)])
        lead = 3 * SR  # media plays 3 s before he speaks (canceller can converge as in real use)
        near = np.concatenate([np.zeros(lead, np.float32), near]); n += lead
        if media:
            m = rd(os.path.join(src, media + '.wav'))
            m = np.tile(m, int(np.ceil(n / len(m))))[:n]
            e = echo(m)
            e = e * db(NEAR_DB - ser) / active_rms(e)
            ref = m * db(-20) / active_rms(m)  # tap = digital playback level (media ~ -20 dBFS rms)
        else:
            e = np.zeros(n, np.float32); ref = np.zeros(n, np.float32)
        noise = rng.standard_normal(n).astype(np.float32) * db(-70)
        d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
        wr(os.path.join(d, base + '.mic.wav'), near + e + noise)
        wr(os.path.join(d, base + '.ref.wav'), ref)
        wr(os.path.join(d, base + '.near.wav'), near)
print('ok', len(clips), 'clips x', len(conds), 'conds')
