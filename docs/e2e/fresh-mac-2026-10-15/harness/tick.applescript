-- tick.applescript <label>: click the checkbox with that title/description anywhere in the Always onboarding window
on run argv
  set want to item 1 of argv
  tell application "System Events"
    tell process "Always"
      set w to (first window whose name contains "Always")
      set els to entire contents of w
      repeat with e in els
        try
          if role of e is "AXCheckBox" then
            set t to ""
            try
              set t to title of e
            end try
            set d to ""
            try
              set d to description of e
            end try
            if t is want or d is want or want is "*first*" then
              click e
              return "ticked " & want & " value=" & (value of e as text)
            end if
          end if
        end try
      end repeat
      set names to ""
      repeat with e in els
        try
          if role of e is "AXCheckBox" then set names to names & "[" & (title of e as text) & "|" & (description of e as text) & "]"
        end try
      end repeat
      return "NOT FOUND " & want & " among " & names
    end tell
  end tell
end run
