#!/bin/bash
# shot.sh <name>: screenshot + window/button inventory
screencapture -x ~/e2e/shots/$1.png
osascript ~/e2e/uiwins.applescript 2>/dev/null
