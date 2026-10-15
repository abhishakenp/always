# asr_whisper.py WAV... : whisper base.en (CPU, fp32, greedy, no prompt carry-over) -> "path\ttext"
# Second, noise-robust oracle; run: nice -n 15 taskpolicy -b <whisper-python> asr_whisper.py ...
import sys, warnings, torch; warnings.filterwarnings('ignore'); torch.set_num_threads(2)
import whisper
m = whisper.load_model('base.en')
for p in sys.argv[1:]:
    r = m.transcribe(p, language='en', fp16=False, temperature=0, condition_on_previous_text=False)
    print(p + '\t' + r['text'].strip().replace('\t', ' '), flush=True)
