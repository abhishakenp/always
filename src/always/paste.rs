//! Clipboard + paste abstraction.
//!
//! The macOS implementation shells out to `pbcopy` for clipboard write and
//! posts synthetic Cmd+V / Return events via Core Graphics. The trait
//! ([`ClipboardProvider`]) exists so the `event_loop` can be exercised
//! end-to-end in tests without actually touching the user's pasteboard,
//! and so future Linux/Windows backends can plug in without rewriting
//! callers.
//!
//! `MockClipboardProvider` (gated behind `cfg(test)`) records every
//! `copy` and `paste` call so tests can assert exactly what would have
//! been sent to the OS.

use anyhow::{Context, Result};

/// Clipboard + simulated-paste capability.
///
/// All implementations must be `Send + Sync` so the daemon can hand a
/// trait object across thread boundaries.
pub trait ClipboardProvider: Send + Sync {
    /// Write `text` to the system clipboard, replacing the previous content.
    fn copy(&self, text: &str) -> Result<()>;

    /// Paste the clipboard at the current keyboard focus, optionally
    /// followed by Return when `auto_enter` is set.
    fn paste(&self, auto_enter: bool) -> Result<()>;

    /// Convenience: copy followed by paste. Default impl is what every
    /// production caller wants; no need to override.
    fn copy_and_paste(&self, text: &str, auto_enter: bool) -> Result<()> {
        self.copy(text)?;
        self.paste(auto_enter)
    }
}

/// macOS-native clipboard via `pbcopy` + Core Graphics keyboard events.
///
/// Stateless — safe to construct anywhere and share via `Arc`.
#[derive(Debug, Default, Clone, Copy)]
pub struct MacClipboard;

impl ClipboardProvider for MacClipboard {
    fn copy(&self, text: &str) -> Result<()> {
        copy_to_clipboard(text.to_string())
    }

    fn paste(&self, auto_enter: bool) -> Result<()> {
        paste_text(auto_enter)
    }
}

/// Sandbox guard (launch smoke test, `ALWAYS_NO_PASTE=1`): a sandboxed
/// daemon must never touch the user's clipboard or type into their apps.
fn no_paste_env() -> bool {
    std::env::var("ALWAYS_NO_PASTE").as_deref() == Ok("1")
}

pub fn copy_to_clipboard(text: String) -> Result<()> {
    if no_paste_env() {
        tracing::info!(chars = text.chars().count(), "clipboard_write_suppressed_no_paste_env");
        return Ok(());
    }
    // Native first: an in-process NSPasteboard write. `pbcopy` costs a
    // fork+exec per paste — measured p50 17.4 ms / p95 22.0 ms (20 runs,
    // 2026-10-13), which was essentially all of `paste_ms`. The subprocess
    // stays as the fallback if the runtime path refuses the write.
    #[cfg(feature = "macos")]
    if native_pasteboard_write(&text) {
        return Ok(());
    }
    copy_via_subprocess(text)
}

/// `[NSPasteboard.generalPasteboard clearContents]` +
/// `setString:forType:"public.utf8-plain-text"` through raw objc-runtime
/// calls (same approach as [`pasteboard_change_count`], no AppKit crate).
/// Returns false when any lookup fails or the pasteboard refuses the write.
#[cfg(feature = "macos")]
fn native_pasteboard_write(text: &str) -> bool {
    use std::ffi::c_void;

    #[link(name = "AppKit", kind = "framework")]
    unsafe extern "C" {}
    unsafe extern "C" {
        fn objc_getClass(name: *const u8) -> *mut c_void;
        fn sel_registerName(name: *const u8) -> *mut c_void;
        fn objc_msgSend();
        fn objc_autoreleasePoolPush() -> *mut c_void;
        fn objc_autoreleasePoolPop(pool: *mut c_void);
    }
    const NS_UTF8_STRING_ENCODING: usize = 4;

    unsafe {
        let pool = objc_autoreleasePoolPush();
        let ok = (|| {
            let send0: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
                std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
            let send_bytes: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *const u8,
                usize,
                usize,
            ) -> *mut c_void = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
            let send_set: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *mut c_void,
                *mut c_void,
            ) -> bool = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());

            let pb_class = objc_getClass(c"NSPasteboard".as_ptr().cast());
            let str_class = objc_getClass(c"NSString".as_ptr().cast());
            if pb_class.is_null() || str_class.is_null() {
                return false;
            }
            let general = send0(pb_class, sel_registerName(c"generalPasteboard".as_ptr().cast()));
            if general.is_null() {
                return false;
            }
            let new_string = |bytes: &[u8]| -> *mut c_void {
                let raw = send0(str_class, sel_registerName(c"alloc".as_ptr().cast()));
                if raw.is_null() {
                    return raw;
                }
                send_bytes(
                    raw,
                    sel_registerName(c"initWithBytes:length:encoding:".as_ptr().cast()),
                    bytes.as_ptr(),
                    bytes.len(),
                    NS_UTF8_STRING_ENCODING,
                )
            };
            let value = new_string(text.as_bytes());
            let kind = new_string(b"public.utf8-plain-text");
            let release = |obj: *mut c_void| {
                if !obj.is_null() {
                    send0(obj, sel_registerName(c"release".as_ptr().cast()));
                }
            };
            if value.is_null() || kind.is_null() {
                release(value);
                release(kind);
                return false;
            }
            send0(general, sel_registerName(c"clearContents".as_ptr().cast()));
            let written = send_set(
                general,
                sel_registerName(c"setString:forType:".as_ptr().cast()),
                value,
                kind,
            );
            release(value);
            release(kind);
            written
        })();
        objc_autoreleasePoolPop(pool);
        if !ok {
            tracing::warn!("native_pasteboard_write_failed_falling_back_to_pbcopy");
        }
        ok
    }
}

fn copy_via_subprocess(text: String) -> Result<()> {
    use std::io::Write as _;
    use std::process::{Command, Stdio};

    let copy_cmd = if cfg!(target_os = "macos") {
        "pbcopy"
    } else if cfg!(target_os = "linux") {
        "xclip"
    } else {
        anyhow::bail!("copy_to_clipboard not implemented for this platform");
    };

    let mut command = Command::new(copy_cmd);
    if cfg!(target_os = "linux") {
        command.args(["-selection", "clipboard", "-in"]);
    }

    let mut copy = command
        .stdin(Stdio::piped())
        .spawn()
        .with_context(|| format!("Failed to run {copy_cmd}"))?;
    let write_res: Result<()> = (|| {
        copy.stdin
            .as_mut()
            .with_context(|| format!("Failed to open {copy_cmd} stdin"))?
            .write_all(text.as_bytes())
            .context("Failed to write transcript to clipboard")?;
        Ok(())
    })();
    // Always reap the child even when the write failed. `wait()` closes our
    // stdin handle first (so the clipboard tool sees EOF and exits); skipping it on the
    // error path would leak a zombie process on this always-on daemon.
    let status = copy
        .wait()
        .with_context(|| format!("Failed waiting for {copy_cmd}"))?;
    write_res?;
    if !status.success() {
        anyhow::bail!("{copy_cmd} failed");
    }
    Ok(())
}

/// How long after the initial paste we may undo+repaste a grammar patch.
/// Beyond this the user may have edited the field and undo is unsafe.
pub const GRAMMAR_PATCH_MAX_AGE: std::time::Duration = std::time::Duration::from_secs(4);

#[cfg(feature = "macos")]
fn post_cmd_key(
    source: &core_graphics::event_source::CGEventSource,
    keycode: core_graphics::event::CGKeyCode,
) -> Result<()> {
    use core_graphics::event::{CGEvent, CGEventFlags, CGEventTapLocation};

    let down = CGEvent::new_keyboard_event(source.clone(), keycode, true)
        .map_err(|_| anyhow::anyhow!("Failed to create key down event"))?;
    down.set_flags(CGEventFlags::CGEventFlagCommand);
    down.post(CGEventTapLocation::HID);

    let up = CGEvent::new_keyboard_event(source.clone(), keycode, false)
        .map_err(|_| anyhow::anyhow!("Failed to create key up event"))?;
    up.set_flags(CGEventFlags::CGEventFlagCommand);
    up.post(CGEventTapLocation::HID);
    Ok(())
}

/// Undo the most recent paste in the focused app (Cmd+Z).
#[cfg(feature = "macos")]
pub fn undo_last_paste() -> Result<()> {
    if no_paste_env() {
        return Ok(());
    }
    use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};

    let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
        .map_err(|_| anyhow::anyhow!("Failed to create CGEventSource"))?;
    // 'z' key
    post_cmd_key(&source, 6)?;
    // Let the target app process undo before we repaste.
    std::thread::sleep(std::time::Duration::from_millis(40));
    Ok(())
}

/// Replace the last paste with `text` via undo + clipboard paste (no Return).
#[cfg(feature = "macos")]
pub fn replace_via_undo(text: &str) -> Result<()> {
    undo_last_paste()?;
    copy_to_clipboard(text.to_string())?;
    paste_text(false)
}

#[cfg(not(feature = "macos"))]
pub fn undo_last_paste() -> Result<()> {
    if no_paste_env() {
        return Ok(());
    }
    anyhow::bail!("undo_last_paste not implemented for this platform")
}

#[cfg(not(feature = "macos"))]
pub fn replace_via_undo(_text: &str) -> Result<()> {
    anyhow::bail!("replace_via_undo not implemented for this platform")
}

#[cfg(feature = "macos")]
pub fn paste_text(auto_enter: bool) -> Result<()> {
    if no_paste_env() {
        return Ok(());
    }
    use core_graphics::event::{CGEvent, CGEventFlags, CGEventTapLocation, CGKeyCode};
    use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};

    let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
        .map_err(|_| anyhow::anyhow!("Failed to create CGEventSource"))?;

    // Simulate Cmd+V
    let v_keycode: CGKeyCode = 9; // 'v' key
    let key_down = CGEvent::new_keyboard_event(source.clone(), v_keycode, true)
        .map_err(|_| anyhow::anyhow!("Failed to create key down event"))?;
    key_down.set_flags(CGEventFlags::CGEventFlagCommand);
    key_down.post(CGEventTapLocation::HID);

    let key_up = CGEvent::new_keyboard_event(source.clone(), v_keycode, false)
        .map_err(|_| anyhow::anyhow!("Failed to create key up event"))?;
    key_up.set_flags(CGEventFlags::CGEventFlagCommand);
    key_up.post(CGEventTapLocation::HID);

    if auto_enter {
        // Longer sleep: the Cmd flag from the V keystrokes above can
        // still register as held when Return posts < 20ms later. In
        // Ghostty (and other apps that bind Cmd+Return) that surfaces
        // as a spurious binding hit instead of a newline. Bumping to
        // 50ms and explicitly clearing flags on the Return events
        // makes the keystroke read as a bare Return everywhere.
        std::thread::sleep(std::time::Duration::from_millis(50));
        let enter_keycode: CGKeyCode = 36; // Return key
        let enter_down = CGEvent::new_keyboard_event(source.clone(), enter_keycode, true)
            .map_err(|_| anyhow::anyhow!("Failed to create enter key down event"))?;
        enter_down.set_flags(CGEventFlags::empty());
        enter_down.post(CGEventTapLocation::HID);

        let enter_up = CGEvent::new_keyboard_event(source, enter_keycode, false)
            .map_err(|_| anyhow::anyhow!("Failed to create enter key up event"))?;
        enter_up.set_flags(CGEventFlags::empty());
        enter_up.post(CGEventTapLocation::HID);
    }

    Ok(())
}

#[cfg(not(feature = "macos"))]
pub fn paste_text(auto_enter: bool) -> Result<()> {
    if no_paste_env() {
        return Ok(());
    }
    if !cfg!(target_os = "linux") {
        anyhow::bail!("paste_text is not implemented for this platform");
    }
    if focused_window_is_terminal().unwrap_or(false) {
        post_xdotool_key(["key", "--clearmodifiers", "ctrl+shift+v"])?;
    } else {
        post_xdotool_key(["key", "--clearmodifiers", "ctrl+v"])?;
    }
    if auto_enter {
        std::thread::sleep(std::time::Duration::from_millis(50));
        post_xdotool_key(["key", "--clearmodifiers", "Return"])?;
    }
    Ok(())
}

pub fn paste(text: &str, auto_enter: bool) -> Result<()> {
    copy_to_clipboard(text.to_string())?;
    paste_text(auto_enter)?;
    Ok(())
}

/// Read the current clipboard text via `pbpaste`.
///
/// Mirrors `correction::read_clipboard` (the Cmd+C capture path already
/// uses this exact pattern). Used to snapshot the user's clipboard before
/// we overwrite it with a transcript so it can be restored afterwards.
///
/// NOTE: this preserves only the **string** flavor of the pasteboard.
/// Non-text flavors (images, RTF, file URLs) on the prior clipboard are
/// not captured and therefore not restored — plain-text round-trip is the
/// high-value case for an always-on dictation daemon.
pub fn read_clipboard_text() -> Result<String> {
    let (cmd, args): (&str, &[&str]) = if cfg!(target_os = "macos") {
        ("pbpaste", &[])
    } else if cfg!(target_os = "linux") {
        ("xclip", &["-selection", "clipboard", "-out"])
    } else {
        anyhow::bail!("read_clipboard_text not implemented for this platform");
    };
    let output = std::process::Command::new(cmd)
        .args(args)
        .output()
        .with_context(|| format!("spawn {cmd}"))?;
    if !output.status.success() {
        anyhow::bail!("{cmd} exited with status {}", output.status);
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// `NSPasteboard.general.changeCount` — a counter the pasteboard server
/// bumps on every write by any process. Capturing it right after our own
/// `pbcopy` gives a token that detects *any* later clipboard write, even
/// one that wrote the identical string or a non-text flavor (both
/// invisible to the `pbpaste` string compare).
///
/// Raw objc-runtime calls rather than an AppKit crate dependency — two
/// message sends are not worth a new dep tree. Returns `None` off-macOS
/// or if the runtime lookup fails, in which case callers fall back to
/// the string-compare guard alone.
#[cfg(feature = "macos")]
pub fn pasteboard_change_count() -> Option<i64> {
    use std::ffi::c_void;

    // Force AppKit into the process so the NSPasteboard class is
    // registered with the runtime — CoreGraphics alone doesn't pull it in.
    #[link(name = "AppKit", kind = "framework")]
    unsafe extern "C" {}
    unsafe extern "C" {
        fn objc_getClass(name: *const u8) -> *mut c_void;
        fn sel_registerName(name: *const u8) -> *mut c_void;
        fn objc_msgSend();
    }

    unsafe {
        let class = objc_getClass(c"NSPasteboard".as_ptr().cast());
        if class.is_null() {
            return None;
        }
        let send_ptr: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let general = send_ptr(
            class,
            sel_registerName(c"generalPasteboard".as_ptr().cast()),
        );
        if general.is_null() {
            return None;
        }
        let send_int: unsafe extern "C" fn(*mut c_void, *mut c_void) -> i64 =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        Some(send_int(
            general,
            sel_registerName(c"changeCount".as_ptr().cast()),
        ))
    }
}

#[cfg(not(feature = "macos"))]
pub fn pasteboard_change_count() -> Option<i64> {
    None
}

/// Restore a previously-snapshotted clipboard string, but only if the
/// current clipboard still holds exactly what we wrote (`just_wrote`).
///
/// This is the guard against clobbering a *fresh* user copy: after the
/// synthetic Cmd+V has been consumed we re-read the clipboard; if it no
/// longer equals the transcript we put there, then the user (or another
/// app, or the passive watcher) copied something new in the meantime —
/// so we leave their clipboard alone. Only when the clipboard is still
/// our own transcript do we put `prev` back.
///
/// `write_token` is the [`pasteboard_change_count`] captured right after
/// our own clipboard write. When available it is the stronger guard: any
/// later write bumps the count — including one that wrote the *same
/// string* or only non-text flavors, both of which the string compare
/// below cannot see. `None` (off-macOS, runtime lookup failed) degrades
/// to the string compare alone.
///
/// Caller is responsible for sleeping long enough that the target app has
/// consumed the paste before invoking this. Best-effort: a failed restore
/// is logged by the caller, never propagated — losing the user's prior
/// clipboard is a worse outcome than skipping the restore, but a failed
/// re-read here should likewise not abort the caller.
///
/// As with [`read_clipboard_text`], only the string flavor is restored;
/// non-text flavors that were on the clipboard before are not preserved.
pub fn restore_clipboard_if_unchanged(
    prev: &str,
    just_wrote: &str,
    write_token: Option<i64>,
) -> Result<()> {
    // Strong guard: the pasteboard server counted a write after ours —
    // whatever is on the clipboard now, it isn't (only) our transcript.
    if let (Some(token), Some(current)) = (write_token, pasteboard_change_count())
        && current != token
    {
        return Ok(());
    }
    // If the clipboard no longer matches the transcript we wrote, a fresh
    // copy happened during the paste window — don't clobber it.
    match read_clipboard_text() {
        Ok(current) if current == just_wrote => copy_to_clipboard(prev.to_string()),
        Ok(_) => Ok(()), // user/app copied something new — leave it.
        // Re-read failed: be conservative and skip the restore rather than
        // risk overwriting an unknown clipboard state.
        Err(e) => Err(e),
    }
}

/// Synthesize a single Return keypress in the focused app. Used to
/// commit auto-enter after a countdown overlay completes (vs.
/// inline in `paste_text(auto_enter=true)`).
///
/// Explicitly clears `CGEventFlags` on both down and up events: without
/// this the Return can inherit lingering Command/Option state from a
/// preceding Cmd+V (or from a modifier the user happens to be holding)
/// and apps like Ghostty interpret the result as Cmd+Return — a
/// configured binding, not a newline.
#[cfg(feature = "macos")]
pub fn press_return() -> Result<()> {
    if no_paste_env() {
        return Ok(());
    }
    use core_graphics::event::{CGEvent, CGEventFlags, CGEventTapLocation, CGKeyCode};
    use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};

    let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
        .map_err(|_| anyhow::anyhow!("Failed to create CGEventSource"))?;
    let enter_keycode: CGKeyCode = 36;
    let enter_down = CGEvent::new_keyboard_event(source.clone(), enter_keycode, true)
        .map_err(|_| anyhow::anyhow!("Failed to create enter key down event"))?;
    enter_down.set_flags(CGEventFlags::empty());
    enter_down.post(CGEventTapLocation::HID);
    let enter_up = CGEvent::new_keyboard_event(source, enter_keycode, false)
        .map_err(|_| anyhow::anyhow!("Failed to create enter key up event"))?;
    enter_up.set_flags(CGEventFlags::empty());
    enter_up.post(CGEventTapLocation::HID);
    Ok(())
}

#[cfg(not(feature = "macos"))]
pub fn press_return() -> Result<()> {
    if no_paste_env() {
        return Ok(());
    }
    if !cfg!(target_os = "linux") {
        anyhow::bail!("press_return not implemented for this platform");
    }
    post_xdotool_key(["key", "--clearmodifiers", "Return"])
}

#[cfg(not(feature = "macos"))]
fn focused_window_is_terminal() -> Result<bool> {
    let output = std::process::Command::new("xdotool")
        .args(["getactivewindow", "getwindowclassname"])
        .output()
        .context("Failed to query active window class with xdotool")?;
    if !output.status.success() {
        anyhow::bail!(
            "xdotool getwindowclassname failed with status {}",
            output.status
        );
    }
    Ok(is_terminal_window_class(
        String::from_utf8_lossy(&output.stdout).trim(),
    ))
}

#[cfg(not(feature = "macos"))]
fn is_terminal_window_class(class_name: &str) -> bool {
    matches!(
        class_name.to_ascii_lowercase().as_str(),
        "alacritty"
            | "com.mitchellh.ghostty"
            | "foot"
            | "gnome-terminal-server"
            | "hyper"
            | "kgx"
            | "kitty"
            | "konsole"
            | "org.wezfurlong.wezterm"
            | "ptyxis"
            | "tabby"
            | "terminator"
            | "tilix"
            | "urxvt"
            | "wezterm"
            | "xfce4-terminal"
            | "xterm"
    )
}

#[cfg(not(feature = "macos"))]
fn post_xdotool_key<const N: usize>(args: [&str; N]) -> Result<()> {
    let status = std::process::Command::new("xdotool")
        .args(args)
        .status()
        .context("Failed to run xdotool")?;
    if !status.success() {
        anyhow::bail!("xdotool failed with status {status}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// Touches the real general pasteboard, so ignored by default. Restores
    /// the prior text afterwards. Run: `cargo test --features local-stt
    /// --lib native_pasteboard_roundtrip -- --ignored --nocapture`.
    #[cfg(feature = "macos")]
    #[test]
    #[ignore]
    fn native_pasteboard_roundtrip() {
        let prior = super::read_clipboard_text().unwrap_or_default();
        let sample = "Hand-back test — ünïcødé … 🎙️ done ".to_string();
        let mut ms = Vec::new();
        for _ in 0..20 {
            let t = std::time::Instant::now();
            assert!(super::native_pasteboard_write(&sample));
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let back = super::read_clipboard_text().unwrap_or_default();
        let _ = super::copy_to_clipboard(prior);
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("native write ms p50={:.3} p95={:.3} max={:.3}", ms[10], ms[18], ms[19]);
        assert_eq!(back, sample);
    }

    use super::*;
    use mock::MockClipboardProvider;

    #[test]
    fn mock_records_copy_and_paste() {
        let p = MockClipboardProvider::new();
        p.copy("hello").unwrap();
        p.paste(false).unwrap();
        assert_eq!(p.copied(), vec!["hello".to_string()]);
        assert_eq!(p.paste_calls(), vec![false]);
    }

    #[test]
    fn copy_and_paste_default_chains_calls() {
        let p = MockClipboardProvider::new();
        p.copy_and_paste("git status", true).unwrap();
        assert_eq!(p.copied(), vec!["git status".to_string()]);
        assert_eq!(p.paste_calls(), vec![true]);
    }

    #[test]
    fn provider_is_object_safe() {
        // Erased trait object — the daemon hands `Box<dyn ClipboardProvider>`
        // around. If this stops compiling we've broken DI.
        let _: Box<dyn ClipboardProvider> = Box::new(MockClipboardProvider::new());
    }

    #[cfg(not(feature = "macos"))]
    #[test]
    fn terminal_window_classes_use_terminal_paste_chord() {
        assert!(is_terminal_window_class("kitty"));
        assert!(is_terminal_window_class("Gnome-Terminal-Server"));
        assert!(is_terminal_window_class("org.wezfurlong.WezTerm"));
        assert!(!is_terminal_window_class("firefox"));
    }
}

#[cfg(test)]
pub mod mock {
    //! In-memory test double for [`ClipboardProvider`].
    //!
    //! Records every call so tests can assert what the daemon would have
    //! pasted into a real application. Not gated behind `pub(crate)`
    //! because the end-to-end integration tests in `tests/` need it too.

    use super::ClipboardProvider;
    use anyhow::Result;
    use parking_lot::Mutex;
    use std::sync::Arc;

    #[derive(Debug, Default, Clone)]
    pub struct MockClipboardProvider {
        calls: Arc<Mutex<MockState>>,
    }

    #[derive(Debug, Default)]
    struct MockState {
        copied: Vec<String>,
        paste_calls: Vec<bool>,
    }

    impl MockClipboardProvider {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn copied(&self) -> Vec<String> {
            self.calls.lock().copied.clone()
        }

        pub fn paste_calls(&self) -> Vec<bool> {
            self.calls.lock().paste_calls.clone()
        }
    }

    impl ClipboardProvider for MockClipboardProvider {
        fn copy(&self, text: &str) -> Result<()> {
            self.calls.lock().copied.push(text.to_string());
            Ok(())
        }
        fn paste(&self, auto_enter: bool) -> Result<()> {
            self.calls.lock().paste_calls.push(auto_enter);
            Ok(())
        }
    }
}

#[cfg(all(test, feature = "macos"))]
mod pasteboard_tests {
    use super::*;

    /// Restores the plain-text clipboard even when a live probe panics.
    ///
    /// This deliberately mirrors the production limitation documented on
    /// `read_clipboard_text`: non-text pasteboard flavors cannot be preserved.
    struct ClipboardTextRestore(String);

    impl Drop for ClipboardTextRestore {
        fn drop(&mut self) {
            let _ = copy_to_clipboard(self.0.clone());
        }
    }

    /// Live UTF-8 boundary probe for the exact `pbcopy`/`pbpaste` path used
    /// by dictation.  It is ignored because it temporarily owns the real
    /// clipboard; the guard restores its prior plain-text contents.
    #[test]
    #[ignore = "touches the real macOS pasteboard"]
    fn pbcopy_round_trips_french_utf8_bytes() {
        let _restore = ClipboardTextRestore(read_clipboard_text().unwrap_or_default());
        let french = "Salut, ça va ? Même l'été à Genève — déjà vu.";

        MacClipboard
            .copy(french)
            .expect("pbcopy accepts valid UTF-8");
        let pasted = read_clipboard_text().expect("pbpaste reads clipboard text");

        assert_eq!(pasted, french, "text must not be decoded as MacRoman");
        assert_eq!(
            pasted.as_bytes(),
            french.as_bytes(),
            "UTF-8 bytes must survive"
        );
    }

    /// Live probe of the objc changeCount bridge. Ignored by default
    /// because it writes to (and then restores) the real pasteboard.
    #[test]
    #[ignore = "touches the real macOS pasteboard"]
    fn change_count_bumps_on_write_and_restore_respects_token() {
        let prev = read_clipboard_text().unwrap_or_default();

        let before = pasteboard_change_count().expect("changeCount available");
        copy_to_clipboard("changecount-probe".into()).expect("pbcopy");
        let token = pasteboard_change_count().expect("changeCount available");
        assert!(token > before, "write must bump changeCount");

        // Simulate a foreign write after ours: the restore must skip.
        copy_to_clipboard("foreign-write".into()).expect("pbcopy");
        restore_clipboard_if_unchanged(&prev, "changecount-probe", Some(token))
            .expect("guarded restore");
        assert_eq!(
            read_clipboard_text().expect("pbpaste"),
            "foreign-write",
            "restore must not clobber a post-write clipboard change"
        );

        // With a fresh token and untouched clipboard, the restore fires.
        copy_to_clipboard("changecount-probe".into()).expect("pbcopy");
        let token2 = pasteboard_change_count().expect("changeCount available");
        restore_clipboard_if_unchanged(&prev, "changecount-probe", Some(token2))
            .expect("guarded restore");
        assert_eq!(
            read_clipboard_text().expect("pbpaste"),
            prev,
            "restore must put the prior clipboard back"
        );
    }
}
