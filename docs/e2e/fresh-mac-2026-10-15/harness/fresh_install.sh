#!/bin/bash
# Wipe every trace of Always (app, prefs, caches, logs, TCC grants) and
# install the staged bundle into /Applications as a brand-new user would
# (drag to Applications), with Homebrew hidden. Does NOT launch it.
# usage: fresh_install.sh <stage dir containing Always.app>
set -u
STAGE=$1
osascript -e 'tell application "Always" to quit' >/dev/null 2>&1
sleep 1
for p in $(pgrep -f "/Applications/Always.app/"); do kill -9 "$p" 2>/dev/null; done
for p in $(pgrep -f "always-daemon|always-capture|always-preview-asr"); do kill -9 "$p" 2>/dev/null; done
sleep 1
sudo rm -rf /Applications/Always.app
rm -rf "$HOME/Library/Application Support/always" "$HOME/Library/Caches/Always" "$HOME/Library/Caches/always" \
       "$HOME/Library/Logs/Always" "$HOME/.always" /tmp/always_apple_stt.log
defaults delete com.always.v3 >/dev/null 2>&1
tccutil reset All com.always.v3 >/dev/null 2>&1
# tccutil leaves denied rows behind on this image: delete them outright.
sqlite3 "$HOME/Library/Application Support/com.apple.TCC/TCC.db" "delete from access where client like '%always%'"
sudo sqlite3 "/Library/Application Support/com.apple.TCC/TCC.db" "delete from access where client like '%always%'"
sudo launchctl kickstart -k system/com.apple.tccd.system
launchctl kickstart -k gui/$(id -u)/com.apple.tccd
# Homebrew hidden: a new user's Mac has none (and no SoX).
if [ -d /opt/homebrew ] && [ ! -e /opt/homebrew.hidden ]; then sudo mv /opt/homebrew /opt/homebrew.hidden; fi
ditto "$STAGE/Always.app" /Applications/Always.app
xattr -dr com.apple.quarantine /Applications/Always.app 2>/dev/null
SwitchAudioSource -s "BlackHole 2ch" -t input >/dev/null
SwitchAudioSource -s "Apple Virtual Sound Device" -t output >/dev/null
echo "installed: $(defaults read /Applications/Always.app/Contents/Info.plist CFBundleShortVersionString) from $STAGE; brew visible: $(command -v brew || echo no); rec: $(command -v rec || echo no)"
