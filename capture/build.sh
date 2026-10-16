#!/bin/bash
# Build the echo-cancelling capture helper (drop-in replacement for `rec`).
# Output: capture/always-capture. Bundled by Always/build.sh into
# Contents/MacOS/ and signed with the app (stable identity, no TCC churn).
# C DSP (vendored SpeexDSP 1.2.1, BSD — speexdsp/COPYING) + aec_core are
# compiled to objects and linked in; `aec_eval` is the offline harness.
set -euo pipefail
cd "$(dirname "$0")"
OBJ=$(mktemp -d)
trap 'rm -rf "$OBJ"' EXIT
CFLAGS=(-O2 -w -target arm64-apple-macos14.0 -include speexdsp/config.h -Ispeexdsp -Ispeexdsp/speex)
for f in aec_core.c tse_core.c speexdsp/mdf.c speexdsp/preprocess.c speexdsp/fftwrap.c speexdsp/smallft.c \
         speexdsp/filterbank.c speexdsp/resample.c; do
  clang "${CFLAGS[@]}" -c "$f" -o "$OBJ/$(basename "${f%.c}").o"
done
swiftc -O -target arm64-apple-macos14.0 -import-objc-header bridge.h -Xcc -include -Xcc speexdsp/config.h \
  -o always-capture always-capture.swift "$OBJ"/*.o -framework Accelerate
if [ "${1:-}" = "--eval" ]; then
  clang "${CFLAGS[@]}" -o aec_eval aec_eval.c "$OBJ"/aec_core.o "$OBJ"/mdf.o "$OBJ"/preprocess.o \
    "$OBJ"/fftwrap.o "$OBJ"/smallft.o "$OBJ"/filterbank.o
  echo "✓ built $(pwd)/aec_eval"
fi
echo "✓ built $(pwd)/always-capture"
