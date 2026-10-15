# dtln.py SIZE TAG IN_TAG REFSCALE_DB FILES... : DTLN-aec (Westhausen 2021, MIT) on <clip>.IN_TAG.wav with
# <clip>.ref.wav as loopback -> <clip>.TAG.wav. Input mic is scaled so speech sits near -26 dBFS (the
# network was trained at normal levels); output scaled back. Prints CPU per audio second.
import sys, time, numpy as np, scipy.io.wavfile as wf, warnings; warnings.filterwarnings('ignore')
from ai_edge_litert.interpreter import Interpreter
size, tag, intag, refdb = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
D = __import__('os').environ.get('DTLN_DIR', 'dtln') + '/'  # breizhn/DTLN-aec pretrained_models (MIT)
def mk(p):
    i = Interpreter(model_path=D + p, num_threads=1); i.allocate_tensors(); return i
rd = lambda p: wf.read(p)[1].astype(np.float32) / 32768
cpu = 0; aud = 0
for mic in sys.argv[5:]:
    b = mic[:-8]
    i1, i2 = mk(f'dtln_aec_{size}_1.tflite'), mk(f'dtln_aec_{size}_2.tflite')
    d1, o1, d2, o2 = i1.get_input_details(), i1.get_output_details(), i2.get_input_details(), i2.get_output_details()
    x = rd(f'{b}.{intag}.wav'); r = rd(b + '.ref.wav')[:len(x)]
    g = 10 ** 29 / 20 if False else 10 ** (29 / 20)  # raw mic level (-55) -> ~-26 dBFS
    x = x * g; r = r * 10 ** (refdb / 20)
    BL, BS = 512, 128; pad = np.zeros(BL - BS, np.float32)
    x = np.concatenate([pad, x, pad]); r = np.concatenate([pad, r, pad])
    s1 = np.zeros(d1[1]['shape'], np.float32); s2 = np.zeros(d2[1]['shape'], np.float32)
    ib = np.zeros(BL, np.float32); lb = np.zeros(BL, np.float32); ob = np.zeros(BL, np.float32); out = np.zeros(len(x), np.float32)
    t0 = time.process_time()
    for k in range((len(x) - (BL - BS)) // BS):
        ib[:-BS] = ib[BS:]; ib[-BS:] = x[k*BS:(k+1)*BS]; lb[:-BS] = lb[BS:]; lb[-BS:] = r[k*BS:(k+1)*BS]
        F = np.fft.rfft(ib).astype(np.complex64)
        i1.set_tensor(d1[0]['index'], np.abs(F).reshape(1, 1, -1).astype(np.float32))
        i1.set_tensor(d1[2]['index'], np.abs(np.fft.rfft(lb)).reshape(1, 1, -1).astype(np.float32))
        i1.set_tensor(d1[1]['index'], s1); i1.invoke()
        mask = i1.get_tensor(o1[0]['index']); s1 = i1.get_tensor(o1[1]['index'])
        est = np.fft.irfft(F * mask).reshape(1, 1, -1).astype(np.float32)
        i2.set_tensor(d2[1]['index'], s2); i2.set_tensor(d2[0]['index'], est); i2.set_tensor(d2[2]['index'], lb.reshape(1, 1, -1))
        i2.invoke(); ob2 = i2.get_tensor(o2[0]['index']); s2 = i2.get_tensor(o2[1]['index'])
        ob[:-BS] = ob[BS:]; ob[-BS:] = 0; ob += np.squeeze(ob2); out[k*BS:(k+1)*BS] = ob[:BS]
    cpu += time.process_time() - t0; aud += (len(x) - 2*len(pad)) / 16000
    y = out[BL - BS: BL - BS + len(x) - 2*len(pad)] / g
    wf.write(f'{b}.{tag}.wav', 16000, (np.clip(y, -1, 1) * 32767).astype(np.int16))
print(f'dtln {size} {tag}: {cpu:.1f} CPU-s for {aud:.0f} s audio = {cpu/aud*60:.2f} CPU-s/min (python+tflite, 1 thread)')
