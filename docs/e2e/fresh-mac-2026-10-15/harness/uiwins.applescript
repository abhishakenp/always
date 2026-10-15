set out to ""
tell application "System Events"
  repeat with p in (every process)
    try
      set ws to windows of p
      repeat with w in ws
        set btns to ""
        try
          repeat with b in (buttons of w)
            set btns to btns & "[" & (name of b as text) & "]"
          end repeat
        end try
        set out to out & (name of p) & " :: " & (name of w as text) & " :: " & btns & linefeed
      end repeat
    end try
  end repeat
end tell
return out
