import sys, glob, numpy as np, scipy.io.wavfile as wf, warnings; warnings.filterwarnings('ignore')
rd=lambda p: wf.read(p)[1].astype(np.float64)/32768
for d in sys.argv[1:]:
    S=[]
    for m in sorted(glob.glob(d+'/*.mic.wav')):
        x=rd(m); n=rd(m[:-8]+'.near.wav'); seg=slice(48000,len(n)-16000); S.append(10*np.log10(np.sum(n[seg]**2)/np.sum((x[seg]-n[seg])**2)))
    print(f"raw mic    {d.split('/')[-1]:11s} near SDR {np.mean(S):5.1f} dB")
