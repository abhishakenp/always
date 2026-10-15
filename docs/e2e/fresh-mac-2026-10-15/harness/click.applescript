-- click.applescript <process> <button> [<other button that must be in the same window>]
on run argv
  set procName to item 1 of argv
  set btnName to item 2 of argv
  set sib to ""
  if (count of argv) > 2 then set sib to item 3 of argv
  tell application "System Events"
    tell process procName
      repeat with w in windows
        try
          if exists button btnName of w then
            if sib is "" or (exists button sib of w) then
              click button btnName of w
              return "clicked " & btnName & " in " & procName
            end if
          end if
        end try
      end repeat
    end tell
  end tell
  return "NOT FOUND " & btnName & " in " & procName
end run
