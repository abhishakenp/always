#!/usr/bin/env python3
"""Dictation end-to-end harness for a fresh macOS VM.

Plays each synthesized sentence into the BlackHole loopback ("the
microphone"), with TextEdit focused, and watches what Always types there.

usage: dictate.py <label> <voice prefix s|d> [n]
Writes results/<label>-<voice>.jsonl and prints a summary (WER, latency).
Latency = wall time from the END of the spoken sentence to the first text
appearing in TextEdit (what a user perceives as "it typed").
"""
import json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
label, voice = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
refs = [l.strip() for l in open(os.path.join(HERE, "sentences.txt")) if l.strip()][:n]
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
out = open(os.path.join(HERE, "results", f"{label}-{voice}.jsonl"), "w")


def osa(script):
    return subprocess.run(["osascript", "-e", script], capture_output=True, text=True).stdout.rstrip("\n")


def textedit_text():
    return osa('tell application "TextEdit" to get text of document 1')


def reset_doc():
    osa('tell application "TextEdit"\nif (count of documents) = 0 then make new document\nset text of document 1 to ""\nactivate\nend tell')


def words(s):
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return s.split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)], len(r)


errs = total = 0
lats = []
for i, ref in enumerate(refs, 1):
    reset_doc()
    time.sleep(1.0)
    clip = os.path.join(HERE, "corpus", f"{voice}{i}.aiff")
    # AX watcher (20 ms) runs across playback; gives up 12 s after the
    # sentence would have ended, or 2.5 s after the text stops changing.
    w = subprocess.Popen([os.path.join(HERE, "watchtext"), str(time.time() + 20), "2.5"],
                         stdout=subprocess.PIPE, text=True)
    p = subprocess.run([os.path.join(HERE, "playdev"), "BlackHole 2ch", clip], capture_output=True, text=True)
    t0, t1 = map(float, p.stdout.split())
    wo = json.loads(w.communicate()[0])
    first = wo.get("first")
    prev = wo.get("text", "")
    hyp = prev.strip()
    e, nref = wer(ref, hyp)
    errs += e
    total += nref
    lat = (first - t1) if first else None
    if lat is not None:
        lats.append(lat)
    rec = {"i": i, "ref": ref, "hyp": hyp, "word_errors": e, "ref_words": nref,
           "speech_start": t0, "speech_end": t1, "first_text": first,
           "end_to_text_s": None if lat is None else round(lat, 3)}
    out.write(json.dumps(rec) + "\n")
    out.flush()
    print(f"[{i}] {'%.2fs' % lat if lat is not None else 'NO TEXT'}  err={e}/{nref}  {hyp!r}", flush=True)
    time.sleep(1.0)

lats.sort()
med = lats[len(lats) // 2] if lats else None
summary = {"label": label, "voice": voice, "sentences": len(refs), "typed": len(lats),
           "wer": round(errs / total, 4) if total else None,
           "median_end_to_text_s": None if med is None else round(med, 3),
           "max_end_to_text_s": None if not lats else round(lats[-1], 3)}
print(json.dumps(summary))
open(os.path.join(HERE, "results", f"{label}-{voice}.summary.json"), "w").write(json.dumps(summary))
