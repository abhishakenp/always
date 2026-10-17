#!/bin/bash
# Build + run the shout harness against the capture/aec_core.c in this tree (or $1 = path to another aec_core.c).
set -euo pipefail
cd "$(dirname "$0")"
SRC="${1:-aec_core.c}"
OBJ=$(mktemp -d); trap 'rm -rf "$OBJ"' EXIT
CF=(-O2 -w -include speexdsp/config.h -I. -Ispeexdsp -Ispeexdsp/speex)
clang "${CF[@]}" -c "$SRC" -o "$OBJ/aec_core.o"
for f in mdf preprocess fftwrap smallft filterbank resample; do clang "${CF[@]}" -c speexdsp/$f.c -o "$OBJ/$f.o"; done
clang "${CF[@]}" -o "$OBJ/eval" agc_shout_eval.c "$OBJ"/*.o -lm
"$OBJ/eval" 0; "$OBJ/eval" 1
