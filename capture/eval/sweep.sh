#!/bin/bash
# sweep.sh SIM "tag:tail supp act" ... : render warm-started canceller variants (no AGC) + objective metrics
sim=$1; shift; A=${AEC_EVAL:-$(dirname "$0")/../aec_eval}
for v in "$@"; do tag=${v%%:*}; set -- ${v#*:}
  for m in $sim/{music,iris,video,far_music,far_iris}/*.mic.wav; do b=${m%.mic.wav}
    nice -n 15 taskpolicy -b $A $m $b.ref.wav $b.$tag.wav $1 $2 $3 - 0 0 $sim/warm >/dev/null 2>&1; done
  nice -n 15 taskpolicy -b uv run -q --with numpy --with scipy python $(dirname "$0")/obj.py $tag $sim/music $sim/iris $sim/video $sim/far_music $sim/far_iris | sed 's/  (raw.*//'
done
