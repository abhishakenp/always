-- btns.applescript [press N]: list buttons of the Always onboarding window's content group (index, position, size); optionally press #N
on run argv
  set out to ""
  tell application "System Events"
    tell process "Always"
      set w to (first window whose name contains "Always")
      set bs to buttons of group 1 of w
      if (count of argv) > 1 then
        perform action "AXPress" of item ((item 2 of argv) as integer) of bs
        return "pressed #" & (item 2 of argv)
      end if
      set i to 0
      repeat with b in bs
        set i to i + 1
        set out to out & i & ": " & (item 1 of (position of b as list)) & "," & (item 2 of (position of b as list)) & " " & (item 1 of (size of b as list)) & "x" & (item 2 of (size of b as list)) & linefeed
      end repeat
    end tell
  end tell
  return out
end run
