#!/bin/bash
# On the My Voice step: record the three samples (normal / quieter / louder)
# by pressing each Record button and speaking (synthesized) into the
# loopback mic. Prints the daemon's enrollment log lines.
# usage: enroll.sh <shot prefix>
P=${1:-run}
cd ~/e2e
[ -f /tmp/enroll1.aiff ] || say -v Samantha -o /tmp/enroll1.aiff "The quick brown fox jumps over the lazy dog while the sun sets behind the hills and the birds sing softly in the trees"
[ -f /tmp/enroll2.aiff ] || say -v Samantha -o /tmp/enroll2.aiff "I usually dictate my messages quietly when I am sitting in a library or a busy coffee shop with friends nearby"
[ -f /tmp/enroll3.aiff ] || say -v Samantha -o /tmp/enroll3.aiff "When the music is loud I have to speak up so that everyone around me can still hear every word that I say"
for k in 1 2 3; do
  osascript btns.applescript press $k >/dev/null
  sleep 1.5
  ./playdev "BlackHole 2ch" /tmp/enroll$k.aiff >/dev/null
  sleep 5
done
./shot.sh $P-04-voice >/dev/null
L=$(ls -t ~/Library/Logs/Always/always.* | head -1)
grep -o '"message":"[a-z_]*enrol[a-z_]*"[^}]*' "$L" | cut -c1-220 | tail -8
