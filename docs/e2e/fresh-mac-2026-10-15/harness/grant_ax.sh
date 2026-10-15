#!/bin/bash
# Simulate the user toggling Always ON in System Settings → Accessibility
# (the one step macOS never lets an app grant itself; toggling needs the
# admin password in the GUI, which this VM's account does not have).
# Rows without csreq in both TCC dbs — the form this image's tccd honours.
for DB in "/Library/Application Support/com.apple.TCC/TCC.db" "$HOME/Library/Application Support/com.apple.TCC/TCC.db"; do
  sudo sqlite3 "$DB" "insert or replace into access (service,client,client_type,auth_value,auth_reason,auth_version,csreq,flags) values ('kTCCServiceAccessibility','com.always.v3',0,2,4,1,NULL,0)"
done
sudo launchctl kickstart -k system/com.apple.tccd.system
launchctl kickstart -k gui/$(id -u)/com.apple.tccd
echo "accessibility granted (simulated System Settings toggle)"
