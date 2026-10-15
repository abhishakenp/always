# asr.py WAV... : transcribe 16k mono s16 wavs with the INSTALLED always-preview-asr
# (Parakeet EOU, same engine the daemon uses for preview). Prints "path\ttext".
import os, sys, subprocess, struct, json, wave, time
BIN=os.environ.get('PREVIEW_ASR', '/Applications/Always.app/Contents/MacOS/always-preview-asr')  # use a staged copy; run under nice/taskpolicy -b
p=subprocess.Popen([BIN],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
def rd():
    while True:
        l=p.stdout.readline()
        if not l: raise SystemExit('asr died')
        try: j=json.loads(l)
        except: continue
        return j
while rd().get('ev')!='ready': pass
def fr(op,seq,payload=b''):
    p.stdin.write(op+struct.pack('<II',seq,len(payload))+payload); p.stdin.flush()
for seq,path in enumerate(sys.argv[1:],1):
    w=wave.open(path); assert w.getframerate()==16000 and w.getnchannels()==1 and w.getsampwidth()==2, path
    data=w.readframes(w.getnframes())
    fr(b'B',seq)
    for i in range(0,len(data),5120): fr(b'A',seq,data[i:i+5120])
    fr(b'E',seq)
    while True:
        j=rd()
        if j.get('ev')=='final' and j.get('seq')==seq: break
        if j.get('ev')=='error': j={'text':'<ERR '+j.get('msg','')+'>'}; break
    print(path+'\t'+j.get('text',''),flush=True)
p.stdin.close()
