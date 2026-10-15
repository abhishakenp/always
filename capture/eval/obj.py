# obj.py TAG DIR... : objective metrics for <clip>.TAG.wav (rendered WITHOUT AGC, pregain 0):
#  ERLE in echo-only lead (1-3 s) and after-speech tail; near-end SDR during his speech
#  (best lag 0..200 samples; SDR = near energy / error energy, scalar-fitted).
import sys, glob, numpy as np, scipy.io.wavfile as wf, warnings; warnings.filterwarnings('ignore')
rd=lambda p: wf.read(p)[1].astype(np.float64)/32768
tag=sys.argv[1]
for d in sys.argv[2:]:
    E1,E2,S=[],[],[]
    for m in sorted(glob.glob(d+'/*.mic.wav')):
        b=m[:-8]; x=rd(m); y=rd(f'{b}.{tag}.wav'); n=rd(b+'.near.wav')
        s,t=16000,48000
        E1.append(10*np.log10(np.mean(x[s:t]**2)/np.mean(y[s:t]**2)))
        E2.append(10*np.log10(np.mean(x[-16000:]**2)/np.mean(y[-16000:]**2)))
        seg=slice(48000,len(n)-16000); best=-99
        nn0=n[seg]; cc=[np.dot(y[seg.start+L:seg.stop+L],nn0) for L in range(0,400,16)]; L0=int(np.argmax(np.abs(cc)))*16
        for lag in range(max(0,L0-16),L0+17):
            yy=y[seg.start+lag:seg.stop+lag]; nn=n[seg]; g=np.dot(yy,nn)/np.dot(nn,nn)
            best=max(best,10*np.log10(np.dot(nn,nn)*g*g/np.sum((yy-g*nn)**2)))
        S.append(best)
    print(f"{tag:10s} {d.split('/')[-1]:11s} ERLE lead {np.mean(E1):5.1f}  tail {np.mean(E2):5.1f}  near SDR {np.mean(S):5.1f} dB  (raw mic SDR for ref below)")
