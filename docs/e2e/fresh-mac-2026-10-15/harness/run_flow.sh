#!/bin/bash
# Full first-run flow on this fresh VM, as a new user, end to end:
# wipe → install (no Homebrew/SoX/API key) → launch → onboarding (mic,
# accessibility, speech recognition, model, My Voice, app allowlist) →
# dictate 10 sentences into TextEdit.
# usage: run_flow.sh <stage dir> <label>
set -u
STAGE=$1; L=$2
cd ~/e2e
step() { echo "== $*"; }
defaults write com.apple.assistant.support "Dictation Enabled" -bool false   # macOS default
./fresh_install.sh "$STAGE"
osascript -e 'tell application "TextEdit" to quit saving no' >/dev/null 2>&1; sleep 1
osascript -e 'tell application "TextEdit" to activate' -e 'tell application "TextEdit" to make new document' >/dev/null
step "launch"; open /Applications/Always.app; sleep 8; ./shot.sh $L-01-launch
osascript click.applescript UserNotificationCenter Allow "Don’t Allow"
step "accessibility (simulated toggle + relaunch)"; ./grant_ax.sh
osascript -e 'tell application "Always" to quit'; sleep 2; open /Applications/Always.app; sleep 7
osascript btns.applescript press 1 >/dev/null; sleep 2            # Welcome → Continue
step "speech recognition"; osascript btns.applescript press 1 >/dev/null; sleep 2
osascript click.applescript UserNotificationCenter Allow "Don’t Allow"; sleep 7
./shot.sh $L-02-perms-dictation-off >/dev/null
step "user turns on Dictation, Retry"; defaults write com.apple.assistant.support "Dictation Enabled" -bool true
osascript btns.applescript press 2 >/dev/null; sleep 4; ./shot.sh $L-03-perms >/dev/null
n=$(osascript btns.applescript | grep -c :); osascript btns.applescript press $n >/dev/null; sleep 2   # Continue
step "my voice"; ./enroll.sh $L | tail -3
n=$(osascript btns.applescript | grep -c :); osascript btns.applescript press $n >/dev/null; sleep 1.5  # Continue
n=$(osascript btns.applescript | grep -c :); osascript btns.applescript press $n >/dev/null; sleep 2    # Tips → Ready
./shot.sh $L-05-ready >/dev/null
step "allowlist"; osascript tick.applescript "*first*"; sleep 1; ./shot.sh $L-06-ready-ticked >/dev/null
n=$(osascript btns.applescript | grep -c :); osascript btns.applescript press $n >/dev/null; sleep 2    # Start
osascript -e 'tell application "TextEdit" to activate'; sleep 2
step "dictation"; ./dictate.py $L s 10
./shot.sh $L-07-after >/dev/null
