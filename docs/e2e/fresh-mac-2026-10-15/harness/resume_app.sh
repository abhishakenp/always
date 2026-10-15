#!/bin/bash
# Same daemon command the menu bar's "Resume Always for <App>" sends.
python3 -I ~/e2e/udssend.py "$HOME/Library/Caches/Always/always.sock" "{\"type\":\"SetAppPaused\",\"data\":{\"bundle_id\":\"${1:-com.apple.TextEdit}\",\"paused\":false}}"
echo "resumed ${1:-com.apple.TextEdit}"
