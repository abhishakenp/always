//! Capture-consumer stall watchdog.
//!
//! The reader thread reports how many frames wait for the event-loop
//! thread (the only consumer, see `vad::record_utterance`) after each one
//! it queues. A backlog of [`STALL_FRAMES`] or more is a stall: speech
//! queues behind it and the listening cue/overlay comes late (`voice_detected
//! audio_delivery_lag_ms` 6066 at 2026-10-18 12:00:13Z, 16013 at 05:28:45Z).
//!
//! This module only observes: one `capture_consumer_stall` line per stall
//! (the event-loop phase it is stuck in, the backlog) and, for a stall
//! of [`SAMPLE_AFTER_FRAMES`] or more, one `/usr/bin/sample` of the daemon (at
//! most every [`SAMPLE_EVERY`]) into the support dir, so the blocking call
//! is named by its stack rather than guessed.
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Audio queued for the consumer at which it counts as stalled (300 ms).
const STALL_FRAMES: usize = 10;
const SAMPLE_AFTER_FRAMES: usize = 34;
const SAMPLE_EVERY: Duration = Duration::from_secs(600);

static QUEUED: AtomicUsize = AtomicUsize::new(0);
static PHASE: AtomicU8 = AtomicU8::new(0);

/// Where the event-loop thread is. Set at coarse step boundaries only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Phase {
    Unknown = 0,
    LoopTop = 1,
    Housekeeping = 2,
    Gated = 3,
    Recording = 4,
    AfterRecord = 5,
    HandleSpeech = 6,
    HandleDrop = 7,
    /// `record_utterance` past its frame loop (gate verdict, STT, trims).
    Transcribing = 8,
}

impl Phase {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::LoopTop,
            2 => Self::Housekeeping,
            3 => Self::Gated,
            4 => Self::Recording,
            5 => Self::AfterRecord,
            6 => Self::HandleSpeech,
            7 => Self::HandleDrop,
            8 => Self::Transcribing,
            _ => Self::Unknown,
        }
    }
}

pub fn set_phase(p: Phase) {
    PHASE.store(p as u8, Ordering::Relaxed);
}

/// Reader thread, after each queued frame: frames now waiting for the
/// consumer. Near 1 while it keeps up; a backlog is exactly the lag
/// `voice_detected audio_delivery_lag_ms` reports later.
pub fn queued(depth: usize) {
    QUEUED.store(depth, Ordering::Relaxed);
}

/// Pure verdict, for tests: the backlog when it counts as a stall.
#[must_use]
pub fn stalled_for(depth: usize) -> Option<Duration> {
    (depth >= STALL_FRAMES).then(|| Duration::from_millis(depth as u64 * 30))
}

/// Start the watchdog thread once. No-op when already running.
pub fn spawn() {
    static STARTED: AtomicBool = AtomicBool::new(false);
    if STARTED.swap(true, Ordering::AcqRel) {
        return;
    }
    let _ = std::thread::Builder::new()
        .name("stall-watch".into())
        .spawn(|| {
            let mut open: Option<(Phase, usize, bool)> = None;
            let mut last_sample: Option<Instant> = None;
            loop {
                std::thread::sleep(Duration::from_millis(50));
                let depth = QUEUED.load(Ordering::Relaxed);
                match (stalled_for(depth), open.as_mut()) {
                    (Some(d), None) => {
                        let phase = Phase::from_u8(PHASE.load(Ordering::Relaxed));
                        tracing::warn!(phase = ?phase, backlog_ms = d.as_millis() as u64, "capture_consumer_stall");
                        open = Some((phase, depth, false));
                    }
                    (Some(_), Some((_, max, sampled))) => {
                        *max = (*max).max(depth);
                        if depth >= SAMPLE_AFTER_FRAMES
                            && !*sampled
                            && last_sample.is_none_or(|t| t.elapsed() >= SAMPLE_EVERY)
                        {
                            *sampled = true;
                            last_sample = Some(Instant::now());
                            sample_self();
                        }
                    }
                    (None, Some((phase, max, sampled))) => {
                        tracing::warn!(
                            phase = ?phase,
                            max_backlog_ms = *max as u64 * 30,
                            sampled = *sampled,
                            "capture_consumer_stall_ended"
                        );
                        open = None;
                    }
                    (None, None) => {}
                }
            }
        });
}

fn sample_self() {
    let dir = crate::config::config_dir().join("stalls");
    let _ = std::fs::create_dir_all(&dir);
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let file = dir.join(format!("stall-{ts}.txt"));
    let pid = std::process::id().to_string();
    match std::process::Command::new("/usr/bin/sample")
        .args([pid.as_str(), "1", "-mayDie", "-file"])
        .arg(&file)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(_) => tracing::warn!(file = %file.display(), "capture_consumer_stall_sampled"),
        Err(e) => tracing::warn!(error = %e, "capture_consumer_stall_sample_failed"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stall_is_a_backlog_of_300_ms_or_more() {
        assert_eq!(stalled_for(0), None);
        assert_eq!(stalled_for(1), None);
        assert_eq!(stalled_for(9), None);
        assert_eq!(stalled_for(10), Some(Duration::from_millis(300)));
        assert_eq!(stalled_for(200), Some(Duration::from_millis(6000)));
    }
}
