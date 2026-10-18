//! Capture-consumer stall watchdog.
//!
//! The reader thread stamps every frame it queues; the event-loop thread
//! (the only consumer, see `vad::record_utterance`) stamps every frame it
//! takes. While the helper keeps delivering and the consumer has not taken
//! a frame for [`STALL_AFTER`], the consumer is stalled: speech queues
//! behind it and the listening cue/overlay comes late (`voice_detected
//! audio_delivery_lag_ms` 6066 at 2026-10-18 12:00:13Z, 16013 at 05:28:45Z).
//!
//! This module only observes: one `capture_consumer_stall` line per stall
//! (the event-loop phase it is stuck in, the backlog) and, for a stall
//! longer than [`SAMPLE_AFTER`], one `/usr/bin/sample` of the daemon (at
//! most every [`SAMPLE_EVERY`]) into the support dir, so the blocking call
//! is named by its stack rather than guessed.
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::time::{Duration, Instant};

const STALL_AFTER: Duration = Duration::from_millis(300);
const SAMPLE_AFTER: Duration = Duration::from_millis(1000);
const SAMPLE_EVERY: Duration = Duration::from_secs(600);

static PUSHED_US: AtomicU64 = AtomicU64::new(0);
static TAKEN_US: AtomicU64 = AtomicU64::new(0);
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

fn clock() -> &'static Instant {
    static BASE: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    BASE.get_or_init(Instant::now)
}

fn now_us() -> u64 {
    clock().elapsed().as_micros() as u64 + 1
}

pub fn set_phase(p: Phase) {
    PHASE.store(p as u8, Ordering::Relaxed);
}

/// Reader thread: a frame was queued.
pub fn frame_pushed() {
    PUSHED_US.store(now_us(), Ordering::Relaxed);
}

/// Consumer: a frame was taken (or the queue was drained on purpose).
pub fn frame_taken() {
    TAKEN_US.store(now_us(), Ordering::Relaxed);
}

/// Pure verdict, for tests: stalled for how long, if at all.
#[must_use]
pub fn stalled_for(now_us: u64, pushed_us: u64, taken_us: u64) -> Option<Duration> {
    // Nothing taken yet = still starting up (model loads), not a stall.
    if pushed_us == 0 || taken_us == 0 || pushed_us <= taken_us {
        return None;
    }
    let waited = Duration::from_micros(now_us.saturating_sub(taken_us.max(1)));
    // Frames are arriving (pushed within the stall window) but none taken.
    let fresh = now_us.saturating_sub(pushed_us) < STALL_AFTER.as_micros() as u64;
    (fresh && waited >= STALL_AFTER).then_some(waited)
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
            let mut open: Option<(u64, Phase, bool)> = None;
            let mut last_sample: Option<Instant> = None;
            loop {
                std::thread::sleep(Duration::from_millis(50));
                let (now, pushed, taken) = (
                    now_us(),
                    PUSHED_US.load(Ordering::Relaxed),
                    TAKEN_US.load(Ordering::Relaxed),
                );
                match (stalled_for(now, pushed, taken), open.as_mut()) {
                    (Some(d), None) => {
                        let phase = Phase::from_u8(PHASE.load(Ordering::Relaxed));
                        tracing::warn!(phase = ?phase, stalled_ms = d.as_millis() as u64, "capture_consumer_stall");
                        open = Some((taken, phase, false));
                    }
                    (Some(d), Some((_, _, sampled))) => {
                        if d >= SAMPLE_AFTER
                            && !*sampled
                            && last_sample.is_none_or(|t| t.elapsed() >= SAMPLE_EVERY)
                        {
                            *sampled = true;
                            last_sample = Some(Instant::now());
                            sample_self();
                        }
                    }
                    (None, Some((since, phase, sampled))) => {
                        tracing::warn!(
                            phase = ?phase,
                            stalled_ms = now.saturating_sub(*since) / 1000,
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
    fn stall_only_while_frames_arrive_untaken() {
        let ms = 1000;
        // Taken just now: no stall.
        assert_eq!(stalled_for(10_000 * ms, 10_000 * ms, 10_000 * ms), None);
        // Pushed now, last taken 400 ms ago: stalled 400 ms.
        assert_eq!(
            stalled_for(10_000 * ms, 10_000 * ms, 9_600 * ms),
            Some(Duration::from_millis(400))
        );
        // Under the bar.
        assert_eq!(stalled_for(10_000 * ms, 10_000 * ms, 9_800 * ms), None);
        // Helper silent (nothing pushed for 2 s): not a consumer stall.
        assert_eq!(stalled_for(10_000 * ms, 8_000 * ms, 7_000 * ms), None);
        // Nothing ever pushed / nothing taken yet (startup).
        assert_eq!(stalled_for(10_000 * ms, 0, 0), None);
        assert_eq!(stalled_for(10_000 * ms, 10_000 * ms, 0), None);
    }
}
