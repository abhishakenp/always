#!/bin/bash
# Synthesise the test sentences with two system voices (US + UK).
cd "$(dirname "$0")"; mkdir -p corpus
i=0
while IFS= read -r line; do
  i=$((i+1))
  say -v Samantha -o corpus/s$i.aiff "$line"
  say -v Daniel -o corpus/d$i.aiff "$line"
done < sentences.txt
ls corpus | wc -l
