# wer.py TRUTH.tsv HYP.tsv : WER per condition/pipeline. TSV lines: path\ttext.
# Truth keyed by clip basename; hyp path .../<cond>/<clip>.<pipe>.wav
import sys, re, os, collections
def norm(t): return re.sub(r"[^a-z0-9' ]", ' ', t.lower()).split()
def ed(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j-1] + 1, p + (x != y))
    return d[len(b)]
truth = {}
for l in open(sys.argv[1]):
    p, t = l.rstrip('\n').split('\t', 1); truth[os.path.basename(p).split('.')[0]] = norm(t)
agg = collections.defaultdict(lambda: [0, 0])
for l in open(sys.argv[2]):
    p, t = l.rstrip('\n').split('\t', 1)
    cond = os.path.basename(os.path.dirname(p)); clip, pipe = os.path.basename(p).split('.')[:2]
    r = truth[clip]; agg[(cond, pipe)][0] += ed(r, norm(t)); agg[(cond, pipe)][1] += len(r)
conds = sorted({c for c, _ in agg}); pipes = sorted({p for _, p in agg})
print('cond'.ljust(12) + ''.join(p.rjust(10) for p in pipes))
for c in conds:
    print(c.ljust(12) + ''.join((f"{100*agg[(c,p)][0]/agg[(c,p)][1]:9.1f}%" if (c,p) in agg else ' '*10) for p in pipes))
