//! Graceful shutdown: SIGTERM never loses the sentence he is saying.
//!
//! 2026-10-16: an Always restart cut an utterance mid-sentence; its only
//! trace was Iris's notch preview. Now SIGTERM/SIGINT ask the capture loop
//! to stop instead of `exit(0)` at once:
//!
//! 1. `request()` — the capture loop (`vad::record_utterance`) checks
//!    [`requested`] on every frame. Idle → it returns at once. Mid-utterance
//!    → it cuts there, as if the endpoint fired, stashing a copy of the
//!    buffered audio ([`begin_cut`]), and the normal path runs: STT, then
//!    paste — or, in consume mode, the held hand-off to Iris.
//! 2. The signal task waits up to [`DRAIN`] for the event loop to report
//!    it stopped ([`loop_stopped`]) and for no held final to be pending.
//! 3. [`finish`] — whatever did not go out (STT failed, the drain ran out,
//!    a held final nobody answered) is kept: the learning ring entry is
//!    written now (`learn::commit_staged_now`, verified voice only), and a
//!    recover file pair is written to `<config_dir>/recover/<ms>.{wav,json}`
//!    (0700 dir, 0600 files) with the audio, the live partial and any text.
//!    It is logged (`shutdown_utterance_recovered`) and listed at the next
//!    start (`shutdown_recover_pending`).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Longest a SIGTERM waits for the utterance in progress to go out.
pub const DRAIN: Duration = Duration::from_millis(3_000);

static REQUESTED: AtomicBool = AtomicBool::new(false);
static LOOP_STOPPED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Default, Clone)]
struct Cut {
    samples: Vec<i16>,
    partial: Option<String>,
    text: Option<String>,
    delivered: bool,
    at_unix_ms: u64,
}

static CUT: Mutex<Option<Cut>> = Mutex::new(None);

fn cut_lock() -> std::sync::MutexGuard<'static, Option<Cut>> {
    CUT.lock().unwrap_or_else(|p| p.into_inner())
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// The signal arrived: finish what is in progress, start nothing new.
pub fn request() {
    REQUESTED.store(true, Ordering::SeqCst);
}

pub fn requested() -> bool {
    REQUESTED.load(Ordering::SeqCst)
}

/// Capture loop: the utterance was cut by the shutdown. Keeps a copy of
/// its audio until it is known to have gone out.
pub fn begin_cut(samples: &[i16], partial: Option<String>) {
    *cut_lock() = Some(Cut {
        samples: samples.to_vec(),
        partial: partial.filter(|p| !p.trim().is_empty()),
        text: None,
        delivered: false,
        at_unix_ms: unix_ms(),
    });
}

/// STT produced the final text of the cut utterance.
pub fn set_text(text: &str) {
    if let Some(c) = cut_lock().as_mut() {
        c.text = Some(text.to_string());
    }
}

/// The cut utterance was pasted or handed to the consumer.
pub fn delivered() {
    if let Some(c) = cut_lock().as_mut() {
        c.delivered = true;
    }
}

/// Event loop: it has stopped taking utterances.
pub fn loop_stopped() {
    LOOP_STOPPED.store(true, Ordering::SeqCst);
}

pub fn is_loop_stopped() -> bool {
    LOOP_STOPPED.load(Ordering::SeqCst)
}

/// True once nothing is left in flight: the loop stopped and no held final waits.
pub fn drained() -> bool {
    is_loop_stopped() && !crate::always::held::busy()
}

/// What the shutdown did with the utterance in progress. Logged by the caller.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Report {
    /// The drain finished inside the budget.
    pub drained: bool,
    /// An utterance was cut by the shutdown.
    pub cut: bool,
    /// It was pasted / handed over before exit.
    pub delivered: bool,
    /// Learning-ring id written now (not pasted, verified voice).
    pub learn_id: Option<String>,
    /// Recover file (json) written for what did not go out.
    pub recover: Option<PathBuf>,
    /// Held finals nobody answered before exit.
    pub held_left: usize,
}

pub fn recover_dir() -> PathBuf {
    crate::config::config_dir().join("recover")
}

/// Keep what did not go out. Never blocks long: local file writes only,
/// plus a bounded wait for the learning-ring writer.
pub fn finish(drained: bool) -> Report {
    finish_into(drained, &recover_dir())
}

fn finish_into(drained: bool, dir: &Path) -> Report {
    let mut report = Report { drained, ..Default::default() };
    let cut = cut_lock().take();
    let held = crate::always::held::pending_texts();
    report.held_left = held.len();
    if let Some(c) = &cut {
        report.cut = true;
        report.delivered = c.delivered;
    }
    // Pasted utterances were handed to the ring's writer thread: let it land.
    let _ = crate::always::learn::flush(Duration::from_millis(400));
    let undelivered = cut.as_ref().filter(|c| !c.delivered);
    if undelivered.is_some() {
        report.learn_id = crate::always::learn::commit_staged_now();
    }
    if undelivered.is_some() || !held.is_empty() {
        match write_recover(dir, undelivered, &held) {
            Ok(path) => report.recover = Some(path),
            Err(e) => tracing::warn!(error = %e, "shutdown_recover_write_failed"),
        }
    }
    report
}

fn private_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut o = std::fs::OpenOptions::new();
        o.create(true).write(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            o.mode(0o600);
        }
        let mut f = o.open(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

fn write_recover(dir: &Path, cut: Option<&Cut>, held: &[String]) -> anyhow::Result<PathBuf> {
    private_dir(dir)?;
    let at = cut.map_or_else(unix_ms, |c| c.at_unix_ms);
    let stem = at.to_string();
    let mut wav: Option<String> = None;
    if let Some(c) = cut
        && !c.samples.is_empty()
    {
        let bytes = crate::always::audio::create_wav_bytes_i16_mono_16k(&c.samples)?;
        let path = dir.join(format!("{stem}.wav"));
        write_private(&path, &bytes)?;
        wav = Some(format!("{stem}.wav"));
    }
    let json = serde_json::json!({
        "at_unix_ms": at,
        "reason": "shutdown",
        "secs": cut.map_or(0.0, |c| c.samples.len() as f64 / 16_000.0),
        "wav": wav,
        "partial": cut.and_then(|c| c.partial.clone()),
        "text": cut.and_then(|c| c.text.clone()),
        "held": held,
    });
    let path = dir.join(format!("{stem}.json"));
    write_private(&path, format!("{json}\n").as_bytes())?;
    Ok(path)
}

/// At start: say how many recover files a past shutdown left.
pub fn report_pending_on_start() {
    let dir = recover_dir();
    let Ok(read) = std::fs::read_dir(&dir) else {
        return;
    };
    let n = read
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .count();
    if n > 0 {
        tracing::warn!(count = n, dir = %dir.display(), "shutdown_recover_pending");
    }
}

/// The signal task's wait: until [`drained`] or the budget (or a second
/// signal, via `abort`) runs out. Returns whether it drained.
pub fn wait_drained(budget: Duration, abort: impl Fn() -> bool) -> bool {
    let deadline = Instant::now() + budget;
    loop {
        if drained() {
            return true;
        }
        if Instant::now() >= deadline || abort() {
            return false;
        }
        std::thread::sleep(Duration::from_millis(15));
    }
}

#[cfg(test)]
pub(crate) fn reset_for_test() {
    REQUESTED.store(false, Ordering::SeqCst);
    LOOP_STOPPED.store(false, Ordering::SeqCst);
    *cut_lock() = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test owns the process-global flags (cargo runs tests in threads).
    #[test]
    fn shutdown_keeps_what_did_not_go_out() {
        reset_for_test();
        let dir = std::env::temp_dir().join(format!("always-shutdown-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // Nothing in progress: no recover file.
        let r = finish_into(true, &dir);
        assert!(!r.cut && r.recover.is_none());

        // Cut and delivered: nothing to recover.
        request();
        assert!(requested());
        begin_cut(&[1i16; 1600], Some("so the plan".into()));
        set_text("So the plan is");
        delivered();
        let r = finish_into(true, &dir);
        assert!(r.cut && r.delivered && r.recover.is_none());

        // Cut, STT never finished: audio + partial kept, private.
        begin_cut(&[7i16; 16_000], Some("what we should do with the gate".into()));
        let r = finish_into(false, &dir);
        assert!(r.cut && !r.delivered && !r.drained);
        let json_path = r.recover.expect("recover file");
        let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap()).unwrap();
        assert_eq!(json["partial"], "what we should do with the gate");
        assert_eq!(json["reason"], "shutdown");
        assert!((json["secs"].as_f64().unwrap() - 1.0).abs() < 1e-6);
        let wav = dir.join(json["wav"].as_str().unwrap());
        assert!(std::fs::metadata(&wav).unwrap().len() > 32_000);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&json_path).unwrap().permissions().mode() & 0o777, 0o600);
            assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
        }

        // The wait: drained once the loop reports stopped; bounded otherwise.
        let t = Instant::now();
        assert!(!wait_drained(Duration::from_millis(60), || false));
        assert!(t.elapsed() >= Duration::from_millis(60));
        loop_stopped();
        assert!(is_loop_stopped());

        reset_for_test();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
