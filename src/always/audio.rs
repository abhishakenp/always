use std::collections::VecDeque;
use std::io;
use std::io::Read as _;
use std::path::PathBuf;
use std::process::{Child, ChildStdout};
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use parking_lot::Mutex;

pub const RATE: u32 = 16_000;
pub const FRAME_MS: u32 = 30;
pub const FRAME_SAMPLES: usize = 480;
pub const FRAME_BYTES: usize = 960;

/// CoreAudio overrun RATE that triggers a `rec` respawn: at least this
/// many "unhandled buffer overrun" lines inside one
/// [`REC_OVERRUN_WINDOW_SECS`] window. Sustained bursts mean SoX is
/// discarding input samples — usually a native-rate capture / resample
/// mismatch on USB mics (e.g. Elgato Wave:3 @ 48 kHz stereo).
///
/// Rate-based, NOT lifetime-cumulative: the old lifetime counter never
/// decayed, so one benign backpressure episode (event loop deliberately
/// not draining the pipe) condemned a healthy recorder at the next
/// `get_or_spawn` — measured at 291 respawns in 2 days, each paying the
/// ~4.5 s device cold start (see `READ_FRAME_TIMEOUT_MS`).
const REC_OVERRUN_RESPAWN_THRESHOLD: u32 = 64;

/// Width of the overrun-counting window (see above).
const REC_OVERRUN_WINDOW_SECS: u64 = 60;

/// Max wall-clock wait for the next bytes of a frame before declaring the
/// recorder *wedged* (alive, but permanently producing neither data nor
/// EOF) and recycling it. This exists only to stop a stuck `rec` from
/// holding the global recorder mutex forever and deadlocking every audio
/// caller — it is NOT a real-time-quality knob.
///
/// It must sit well above the device's legitimate inter-frame gap, or it
/// truncates live speech. Real USB capture is burstier than its nominal
/// 30 ms/frame: SoX delivers resampled output in chunks, and an Elgato
/// Wave:3 MK.2 (48 kHz → 16 kHz) was measured with ~4.5 s cold-start and
/// up to ~4.1 s gaps *mid-stream while healthy*. 15 s is ~3× that worst
/// case — generous enough never to false-positive on bursty delivery,
/// still finite so a genuinely dead device recovers instead of hanging.
/// (An earlier 2–3 s value reset the recorder mid-utterance and cut
/// speech off; do not lower it below the measured device gap.)
const READ_FRAME_TIMEOUT_MS: i32 = 15_000;

// Compile-time floor: the wedge timeout must clear the measured worst-case
// device gap (~4.5s) with margin, or it truncates live speech. A regression
// once shipped a 2-3s value that cut utterances off — keep this guard.
const _: () = assert!(READ_FRAME_TIMEOUT_MS >= 10_000);

/// Conservative hygiene recycle for a healthy recorder. The old call-count
/// guard (1000 `get_or_spawn` calls ≈ every 100 s while media plays) paid a
/// ~4.5 s device cold start and dropped audio on every recycle to guard
/// against a SoX memory leak nobody has measured. Health checks cover the
/// real failure modes; this 24 h cap is the remaining backstop.
const REC_MAX_LIFETIME: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);

/// Block until `fd` is readable (or hung up) or `timeout_ms` elapses.
/// `Ok(true)` = readable now, `Ok(false)` = timed out. Raw `poll(2)`
/// rather than a `libc`/`nix` dependency — one struct, one extern.
#[cfg(feature = "macos")]
fn wait_readable(fd: i32, timeout_ms: i32) -> io::Result<bool> {
    #[repr(C)]
    struct PollFd {
        fd: i32,
        events: i16,
        revents: i16,
    }
    const POLLIN: i16 = 0x0001;
    unsafe extern "C" {
        fn poll(fds: *mut PollFd, nfds: u32, timeout: i32) -> i32;
    }

    let mut pfd = PollFd {
        fd,
        events: POLLIN,
        revents: 0,
    };
    loop {
        let rc = unsafe { poll(&mut pfd, 1, timeout_ms) };
        if rc < 0 {
            let err = io::Error::last_os_error();
            // Retried with the full timeout again — a rare signal mid-wait
            // over-waits slightly rather than complicating with a deadline.
            if err.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        // POLLHUP/POLLERR also count as "go read": the read will observe
        // EOF/error immediately instead of blocking.
        return Ok(rc > 0);
    }
}

static TEMP_WAV_COUNTER: AtomicU64 = AtomicU64::new(0);

// Global persistent audio recorder to avoid spawning processes repeatedly
#[cfg(feature = "macos")]
static GLOBAL_RECORDER: LazyLock<Arc<Mutex<Option<RecChild>>>> =
    LazyLock::new(|| Arc::new(Mutex::new(None)));

// Memory pool for audio buffers to reduce allocations
static AUDIO_BUFFER_POOL: LazyLock<Arc<Mutex<VecDeque<Vec<i16>>>>> =
    LazyLock::new(|| Arc::new(Mutex::new(VecDeque::with_capacity(10))));

pub struct AudioBuffer {
    buffer: Vec<i16>,
    pool: Arc<Mutex<VecDeque<Vec<i16>>>>,
}

impl AudioBuffer {
    pub fn get() -> Self {
        let pool = Arc::clone(&AUDIO_BUFFER_POOL);
        let buffer = {
            let mut pool_lock = pool.lock();
            pool_lock
                .pop_front()
                .unwrap_or_else(|| Vec::with_capacity(16000)) // 1 second at 16kHz
        };
        Self { buffer, pool }
    }

    #[allow(clippy::should_implement_trait)] // `as_mut` returns the inner Vec, not a generic AsMut
    pub fn as_mut(&mut self) -> &mut Vec<i16> {
        &mut self.buffer
    }

    pub fn as_slice(&self) -> &[i16] {
        &self.buffer
    }
}

impl Drop for AudioBuffer {
    fn drop(&mut self) {
        // Return buffer to pool. parking_lot is poison-free, so this can never panic.
        self.buffer.clear();
        let mut pool_lock = self.pool.lock();
        if pool_lock.len() < 10 {
            pool_lock.push_back(std::mem::take(&mut self.buffer));
        }
    }
}

/// Tumbling-window overrun counter shared between the stderr drainer
/// (writer) and `is_healthy` (reader). A window older than
/// [`REC_OVERRUN_WINDOW_SECS`] restarts from 1 on the next overrun, so
/// only a *sustained* storm crosses the respawn threshold.
#[cfg(feature = "macos")]
struct OverrunWindow {
    /// Epoch seconds of the current window's first overrun.
    window_start: AtomicU64,
    count: AtomicU32,
}

#[cfg(feature = "macos")]
impl OverrunWindow {
    fn new() -> Self {
        Self {
            window_start: AtomicU64::new(0),
            count: AtomicU32::new(0),
        }
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Record one overrun; returns the count within the current window.
    fn record(&self) -> u32 {
        let now = Self::now_secs();
        let start = self.window_start.load(Ordering::Relaxed);
        if now.saturating_sub(start) >= REC_OVERRUN_WINDOW_SECS {
            self.window_start.store(now, Ordering::Relaxed);
            self.count.store(1, Ordering::Relaxed);
            1
        } else {
            self.count.fetch_add(1, Ordering::Relaxed) + 1
        }
    }

    /// Overrun count within the current window, or 0 once the window
    /// has aged out.
    fn windowed_count(&self) -> u32 {
        let start = self.window_start.load(Ordering::Relaxed);
        if Self::now_secs().saturating_sub(start) >= REC_OVERRUN_WINDOW_SECS {
            return 0;
        }
        self.count.load(Ordering::Relaxed)
    }
}

#[cfg(feature = "macos")]
pub struct RecChild {
    child: Child,
    stdout: ChildStdout,
    reuse_count: u32,
    /// Wall-clock age anchor for the daily hygiene recycle (see
    /// `get_or_spawn`).
    spawned_at: std::time::Instant,
    /// Fed by the stderr drainer on CoreAudio buffer overruns.
    overruns: Arc<OverrunWindow>,
}

/// Executable name of the echo-cancelling capture helper inside
/// `Always.app/Contents/MacOS/` (source: `capture/always-capture.swift`).
pub const CAPTURE_HELPER_NAME: &str = "always-capture";

/// How long a freshly spawned capture helper may take to deliver its first
/// bytes before the recorder falls back to `rec`. VPIO start-up measured
/// ~1.6 s on a MacBook Air (vs ~0.55 s for `rec`); 4 s leaves margin under
/// load without stalling capture for long when VPIO is broken.
#[cfg(feature = "macos")]
const CAPTURE_HELPER_FIRST_BYTES_MS: i32 = 4_000;

/// Set after the capture helper failed to start or stay alive; the daemon
/// then uses `rec` for the rest of its life instead of paying the start-up
/// timeout on every respawn. Cleared by a default-input-device change.
#[cfg(feature = "macos")]
static CAPTURE_HELPER_FAILED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Pref `capture_aec` (applied at config load). Default on.
static AEC_PREF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Apply the `capture_aec` preference (NULL/None = on — the user confirmed
/// dictation over a playing video works with it, 2026-10-14). Takes effect
/// at the next recorder spawn. Status cues are rendered through the
/// helper's own voice-processing output while it runs (see
/// `capture_cue_send`), so its ducking of other audio cannot silence them.
pub fn set_aec_pref(enabled: Option<bool>) {
    AEC_PREF.store(enabled.unwrap_or(true), std::sync::atomic::Ordering::Relaxed);
}

/// Acoustic echo cancellation against the Mac's own output (default ON,
/// see `set_aec_pref`).
/// Env `ALWAYS_AEC` overrides the pref either way (`0/false/off`,
/// `1/true/on`); an explicit `ALWAYS_CAPTURE_BIN` (tests, replay) implies on.
pub fn aec_enabled() -> bool {
    match std::env::var("ALWAYS_AEC").as_deref() {
        Ok("0") | Ok("false") | Ok("off") => return false,
        Ok("1") | Ok("true") | Ok("on") => return true,
        _ => {}
    }
    std::env::var_os("ALWAYS_CAPTURE_BIN").is_some()
        || AEC_PREF.load(std::sync::atomic::Ordering::Relaxed)
}

/// Locate the capture helper: `ALWAYS_CAPTURE_BIN`, then next to the
/// daemon executable (the app bundle's `Contents/MacOS`), then the repo's
/// build output (dev/test runs from `target/`).
pub fn locate_capture_helper() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ALWAYS_CAPTURE_BIN") {
        let p = PathBuf::from(p);
        return p.is_file().then_some(p);
    }
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|d| d.to_path_buf()))
    {
        let p = dir.join(CAPTURE_HELPER_NAME);
        if p.is_file() {
            return Some(p);
        }
    }
    let dev = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("capture")
        .join(CAPTURE_HELPER_NAME);
    dev.is_file().then_some(dev)
}

#[cfg(feature = "macos")]
impl RecChild {
    /// Spawn the recorder: the echo-cancelling capture helper when AEC is
    /// enabled and the helper is installed and delivers audio, otherwise
    /// SoX `rec`. Either way exactly ONE recorder process exists (I4) — a
    /// helper that fails its start-up check is killed and reaped before
    /// `rec` is spawned.
    pub fn spawn() -> Result<Self> {
        if aec_enabled()
            && !CAPTURE_HELPER_FAILED.load(Ordering::Relaxed)
            && let Some(helper) = locate_capture_helper()
        {
            match Self::spawn_capture_helper(&helper) {
                Ok(rec) => return Ok(rec),
                Err(error) => {
                    CAPTURE_HELPER_FAILED.store(true, Ordering::Relaxed);
                    tracing::warn!(error = %error, "capture_helper_fallback_to_rec");
                }
            }
        }
        Self::spawn_rec()
    }

    /// Spawn `always-capture` (VPIO echo cancellation) and require its
    /// first bytes within [`CAPTURE_HELPER_FIRST_BYTES_MS`].
    fn spawn_capture_helper(path: &std::path::Path) -> Result<Self> {
        use std::os::fd::AsRawFd as _;
        tracing::info!(path = %path.display(), "capture_helper_spawn_starting");
        let started = std::time::Instant::now();
        let mut child = std::process::Command::new(path)
            // stdin carries status-cue commands (`play <path>`): cues are
            // rendered through the helper's voice-processing output so they
            // are never ducked by it (see `capture_cue_play`).
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .with_context(|| format!("failed to run '{}'", path.display()))?;
        let stdout = child.stdout.take().context("capture helper stdout missing")?;
        let cue_stdin = child.stdin.take();
        if let Some(stderr) = child.stderr.take() {
            std::thread::spawn(move || {
                use std::io::{BufRead, BufReader};
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        tracing::info!(line = %trimmed, "capture_helper_stderr");
                    }
                }
            });
        }
        let ready = wait_readable(stdout.as_raw_fd(), CAPTURE_HELPER_FIRST_BYTES_MS)?;
        // Readable could also be EOF (helper exited, e.g. VPIO unavailable);
        // `try_wait` tells the two apart.
        let exited = child.try_wait().ok().flatten();
        if !ready || exited.is_some() {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!(
                "capture helper produced no audio within {CAPTURE_HELPER_FIRST_BYTES_MS} ms (exit: {exited:?})"
            );
        }
        tracing::info!(
            pid = child.id(),
            first_bytes_ms = started.elapsed().as_millis() as u64,
            "capture_helper_spawned"
        );
        *CAPTURE_CUE_SINK.lock() = cue_stdin.map(|s| (child.id(), s));
        crate::always::status_sound::capture_route_changed();
        Ok(Self {
            child,
            stdout,
            reuse_count: 0,
            spawned_at: std::time::Instant::now(),
            overruns: Arc::new(OverrunWindow::new()),
        })
    }

    fn spawn_rec() -> Result<Self> {
        tracing::info!("rec_spawn_starting");
        let overruns = Arc::new(OverrunWindow::new());
        let rec_path = if cfg!(target_os = "macos") {
            "/opt/homebrew/bin/rec"
        } else {
            "/usr/bin/rec"
        };
        let mut child = std::process::Command::new(rec_path)
            // Capture at the device's native rate/channels (typically 48 kHz
            // stereo on USB mics), then resample to 16 kHz mono on the
            // output side. Requesting `-c 1 -r 16000` on input makes
            // CoreAudio warn and misbehave ("can't set sample rate 16000;
            // using 48000") which leads to buffer overruns and dropped
            // speech energy in the VAD pipeline.
            .args([
                "--no-show-progress",
                // 131072 is a deliberate stall budget, not a leftover:
                // ~4 s of slack inside SoX (output-side, 16 kHz mono
                // i16) plus ~2 s of kernel pipe. The single event-loop
                // thread legitimately stops draining stdout for seconds
                // at a time (gate sleeps, paste path); with a small
                // buffer SoX blocks on write and CoreAudio starts
                // discarding every callback. See `drain_pending` for why
                // the backlog is discarded on resume.
                "--buffer",
                "131072",
                "-t",
                "raw",
                "-e",
                "signed-integer",
                "-b",
                "16",
                "-",
                "remix",
                "-",
                "rate",
                "16000",
                "channels",
                "1",
            ])
            .stdout(std::process::Stdio::piped())
            // Capture stderr instead of dropping it on the floor. SoX
            // emits permission-denial / device-busy errors to stderr;
            // previously those were silently lost which meant a mic TCC
            // denial looked identical to a healthy idle daemon.
            .stderr(std::process::Stdio::piped())
            .spawn()
            .with_context(|| format!("Failed to run '{rec_path}'. Install SoX"))?;
        let stdout = child.stdout.take().context("sox stdout missing")?;
        if let Some(stderr) = child.stderr.take() {
            let overruns = Arc::clone(&overruns);
            // Drain stderr on a background thread; log every non-empty
            // line as a daemon warning so SoX/CoreAudio errors surface
            // in the structured log instead of disappearing.
            std::thread::spawn(move || {
                use std::io::{BufRead, BufReader};
                let reader = BufReader::new(stderr);
                for line in reader.lines().map_while(Result::ok) {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if trimmed.contains("buffer overrun") {
                        // Overruns while capture is deliberately gated are
                        // backpressure we created ourselves — nobody drains
                        // the pipe during a gate, SoX blocks, CoreAudio
                        // discards. Expected, not a device fault: don't
                        // count them toward a respawn.
                        if crate::always::pause::should_gate_capture() {
                            continue;
                        }
                        let count = overruns.record();
                        // Avoid flooding the log — first + every 32nd.
                        if count == 1 || count.is_multiple_of(32) {
                            tracing::warn!(count, line = %trimmed, "rec_coreaudio_overrun");
                        }
                        continue;
                    }
                    tracing::warn!(line = %trimmed, "rec_stderr");
                }
            });
        }
        tracing::info!(pid = child.id(), "rec_spawned");
        Ok(Self {
            child,
            stdout,
            reuse_count: 0,
            spawned_at: std::time::Instant::now(),
            overruns,
        })
    }

    /// Wall-clock age of this recorder process.
    fn age(&self) -> std::time::Duration {
        self.spawned_at.elapsed()
    }

    pub fn get_or_spawn() -> Result<Arc<Mutex<Option<RecChild>>>> {
        let recorder_lock = Arc::clone(&GLOBAL_RECORDER);
        let mut recorder = recorder_lock.lock();

        let needs_respawn = match recorder.as_mut() {
            Some(rec) => {
                if rec.is_healthy() {
                    rec.reuse_count += 1;
                    // Time-based hygiene cap ONLY. The previous guard
                    // recycled after 1000 *calls* — but the gated loop
                    // calls get_or_spawn every 100 ms tick (wake-on-voice
                    // probe, pipe drain), so a user with media playing
                    // recycled a healthy recorder every ~100 s, each time
                    // paying the ~4.5 s device cold start and dropping
                    // audio mid-window. Real failure modes are covered
                    // below and in `is_healthy`: process death, wedge
                    // timeout, overrun storms, device switches. A daily
                    // quiet recycle remains as a conservative backstop
                    // against slow SoX memory growth.
                    rec.age() > REC_MAX_LIFETIME
                } else {
                    true
                }
            }
            None => true,
        };
        if needs_respawn {
            // Drop (kill + wait) the old recorder BEFORE spawning the
            // replacement, so two `rec` processes never hold the input
            // device at the same time — spawning first briefly ran both
            // against the same CoreAudio device.
            *recorder = None;
            *recorder = Some(Self::spawn()?);
        }
        Ok(Arc::clone(&GLOBAL_RECORDER))
    }

    /// Force-evict the current recorder and spawn a fresh one bound to
    /// whatever macOS now reports as the default input device.
    ///
    /// Triggered by the `RespawnRecorder` UDS command when the Swift app
    /// observes `kAudioHardwarePropertyDefaultInputDevice` change. The
    /// running `rec` opened the input device at spawn time and will not
    /// follow a default-input switch on its own, so without this the
    /// user would have to relaunch Always to use a newly selected mic.
    ///
    /// Same I4 "drop old before spawn new" discipline as
    /// [`get_or_spawn`]: the old `RecChild` is dropped (kill + wait)
    /// before the replacement is spawned, so two `rec` processes never
    /// hold the input device at the same time.
    pub fn force_respawn() -> Result<()> {
        let recorder_lock = Arc::clone(&GLOBAL_RECORDER);
        let mut recorder = recorder_lock.lock();
        if recorder.is_some() {
            tracing::info!("rec_respawn_due_to_default_input_change");
        }
        // A new input device gets a fresh chance at echo cancellation.
        CAPTURE_HELPER_FAILED.store(false, Ordering::Relaxed);
        // Drop (kill + wait) the old recorder BEFORE spawning the
        // replacement — same I4 invariant as `get_or_spawn`.
        *recorder = None;
        *recorder = Some(Self::spawn()?);
        Ok(())
    }

    fn is_healthy(&mut self) -> bool {
        let overruns = self.overruns.windowed_count();
        if overruns >= REC_OVERRUN_RESPAWN_THRESHOLD {
            tracing::warn!(
                overruns,
                threshold = REC_OVERRUN_RESPAWN_THRESHOLD,
                window_secs = REC_OVERRUN_WINDOW_SECS,
                "rec_respawn_due_to_coreaudio_overruns"
            );
            return false;
        }
        match self.child.try_wait() {
            Ok(Some(_)) => false, // Process has exited
            Ok(None) => true,     // Process still running
            Err(_) => false,      // Error checking process
        }
    }

    /// Throw away every byte `rec` has queued up, and report how many
    /// seconds of audio that was.
    ///
    /// `rec` never stops capturing — it runs for the daemon's lifetime
    /// with a 131 KB buffer (~4 s at 16 kHz mono i16), and the kernel
    /// pipe holds more on top. So while capture is gated (mic taken by
    /// another app, user muted, idle-paused) nobody drains that pipe and
    /// it fills with audio from exactly the period Always was supposed
    /// to be deaf. Without this, the first read after resuming returns
    /// that backlog and Always transcribes and pastes speech that
    /// belonged to the other app — the "I unblocked the mic and it
    /// pasted anyway" bug.
    ///
    /// Non-blocking by construction: `wait_readable(fd, 0)` polls with a
    /// zero timeout, so this returns as soon as the pipe runs dry and
    /// can never stall the event loop. A dead recorder (EOF) ends the
    /// loop too — diagnosing that is `read_frame`'s job, not this one.
    pub fn drain_pending(&mut self) -> f64 {
        use std::os::fd::AsRawFd as _;

        let fd = self.stdout.as_raw_fd();
        let mut scratch = [0u8; FRAME_BYTES];
        let mut dropped_bytes = 0usize;
        while wait_readable(fd, 0).unwrap_or(false) {
            match self.stdout.read(&mut scratch) {
                Ok(0) => break, // EOF — recorder died; leave it to read_frame
                Ok(n) => {
                    dropped_bytes += n;
                    capture_heartbeat(&scratch[..n]);
                }
                Err(_) => break,
            }
        }
        // 16 kHz mono i16 = 32 000 bytes per second.
        dropped_bytes as f64 / 32_000.0
    }

    pub fn read_frame(&mut self, buf: &mut [u8; FRAME_BYTES]) -> io::Result<usize> {
        use std::os::fd::AsRawFd as _;

        let fd = self.stdout.as_raw_fd();
        let mut read = 0;
        while read < FRAME_BYTES {
            // Bounded wait before each blocking read so a wedged `rec`
            // (alive, but producing neither bytes nor EOF) can't hold the
            // global recorder mutex forever and deadlock every audio
            // caller. The budget sits well above the device's real
            // (bursty) inter-frame gap so it never truncates live speech —
            // see [`READ_FRAME_TIMEOUT_MS`].
            if !wait_readable(fd, READ_FRAME_TIMEOUT_MS)? {
                tracing::warn!(read, "rec_read_frame_timeout");
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "no audio from rec within timeout — recorder wedged",
                ));
            }
            match self.stdout.read(&mut buf[read..]) {
                Ok(0) => {
                    // EOF on rec's stdout means the recorder died — either
                    // mic permission was denied (TCC), the audio device
                    // was unplugged, or the user killed `rec`. Surface
                    // this once per spawn so we can see it in the daemon
                    // log instead of silently looping on empty reads.
                    if read == 0 {
                        tracing::warn!("rec_eof_on_read_frame");
                    }
                    break;
                }
                Ok(n) => read += n,
                Err(e) => return Err(e),
            }
        }
        capture_heartbeat(&buf[..read]);
        Ok(read)
    }
}

/// The running capture helper's stdin `(pid, pipe)`, for status cues.
static CAPTURE_CUE_SINK: Mutex<Option<(u32, std::process::ChildStdin)>> = Mutex::new(None);

/// Send one line to the capture helper's cue channel. False when no helper
/// is running (plain `rec`) or the pipe is closed — the caller falls back to
/// in-process playback.
pub fn capture_cue_send(line: &str) -> bool {
    use std::io::Write as _;
    let mut sink = CAPTURE_CUE_SINK.lock();
    let Some((_, stdin)) = sink.as_mut() else { return false };
    if stdin.write_all(line.as_bytes()).and_then(|_| stdin.write_all(b"\n")).and_then(|_| stdin.flush()).is_err() {
        *sink = None;
        return false;
    }
    true
}

/// Whether cues currently route through the capture helper.
pub fn capture_cue_available() -> bool {
    CAPTURE_CUE_SINK.lock().is_some()
}

/// Opt-in capture heartbeat (`ALWAYS_CAPTURE_HEARTBEAT=1`, used by the launch
/// smoke test): every 5 s, log how many frames were read and their RMS, so a
/// test can prove audio is actually flowing without needing speech.
fn capture_heartbeat(buf: &[u8]) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("ALWAYS_CAPTURE_HEARTBEAT").as_deref() == Ok("1")) {
        return;
    }
    static FRAMES: AtomicU64 = AtomicU64::new(0);
    static SUM_SQ: parking_lot::Mutex<(f64, u64, Option<std::time::Instant>)> =
        parking_lot::Mutex::new((0.0, 0, None));
    // Counted in bytes (gated drains read partial frames); reported as
    // 30 ms frame equivalents.
    FRAMES.fetch_add(buf.len() as u64, Ordering::Relaxed);
    let mut acc = SUM_SQ.lock();
    for c in buf.chunks_exact(2) {
        let v = i16::from_le_bytes([c[0], c[1]]) as f64 / 32768.0;
        acc.0 += v * v;
        acc.1 += 1;
    }
    let now = std::time::Instant::now();
    let since = *acc.2.get_or_insert(now);
    if now.duration_since(since) >= std::time::Duration::from_secs(5) {
        let rms = (acc.0 / acc.1.max(1) as f64).sqrt();
        tracing::info!(
            frames_total = FRAMES.load(Ordering::Relaxed) / FRAME_BYTES as u64,
            rms,
            "capture_heartbeat"
        );
        *acc = (0.0, 0, Some(now));
    }
}

#[cfg(feature = "macos")]
impl Drop for RecChild {
    fn drop(&mut self) {
        {
            let mut sink = CAPTURE_CUE_SINK.lock();
            if sink.as_ref().is_some_and(|(pid, _)| *pid == self.child.id()) {
                *sink = None;
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Create WAV file data in memory from raw i16 mono samples at 16kHz (optimized)
pub fn create_wav_bytes_i16_mono_16k(samples: &[i16]) -> Result<Vec<u8>> {
    // Pre-calculate size to avoid reallocations
    let data_size = samples.len() * 2; // 2 bytes per i16 sample
    let file_size = 44 + data_size; // WAV header is 44 bytes

    let mut wav_data = Vec::with_capacity(file_size);

    // Write WAV header directly to buffer (faster than using hound for small files)
    wav_data.extend_from_slice(b"RIFF");
    wav_data.extend_from_slice(&((file_size - 8) as u32).to_le_bytes());
    wav_data.extend_from_slice(b"WAVE");
    wav_data.extend_from_slice(b"fmt ");
    wav_data.extend_from_slice(&16u32.to_le_bytes()); // PCM format chunk size
    wav_data.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    wav_data.extend_from_slice(&1u16.to_le_bytes()); // Mono
    wav_data.extend_from_slice(&RATE.to_le_bytes()); // Sample rate
    wav_data.extend_from_slice(&(RATE * 2).to_le_bytes()); // Byte rate
    wav_data.extend_from_slice(&2u16.to_le_bytes()); // Block align
    wav_data.extend_from_slice(&16u16.to_le_bytes()); // Bits per sample
    wav_data.extend_from_slice(b"data");
    wav_data.extend_from_slice(&(data_size as u32).to_le_bytes());

    // Bulk write all samples as little-endian bytes (safe on little-endian platforms)
    // SAFETY: i16 is 2 bytes, alignment of [i16] is 2, alignment of [u8] is 1.
    // Slice from_raw_parts is safe because we don't escape the borrow.
    #[cfg(target_endian = "little")]
    {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(samples.as_ptr() as *const u8, samples.len() * 2) };
        wav_data.extend_from_slice(bytes);
    }
    #[cfg(not(target_endian = "little"))]
    {
        for sample in samples {
            wav_data.extend_from_slice(&sample.to_le_bytes());
        }
    }

    Ok(wav_data)
}

/// Pluggable audio frame source.
///
/// The default production implementation ([`SoxAudioSource`]) shells out
/// to `/opt/homebrew/bin/rec` (SoX) on macOS. The trait exists so:
///
/// 1. Tests can inject a deterministic `mock::MockAudioSource` without
///    touching the user's microphone.
/// 2. Future Linux (`cpal`/ALSA) and Windows (`cpal`/WASAPI) backends can
///    drop in without touching the VAD loop.
pub trait AudioFrameSource: Send {
    /// Read one VAD-sized frame ([`FRAME_BYTES`] bytes of little-endian
    /// 16-bit mono PCM at [`RATE`] Hz). Implementations should fill the
    /// buffer fully or return `Ok(0)` to signal EOF.
    fn read_frame(&mut self, buf: &mut [u8; FRAME_BYTES]) -> io::Result<usize>;
}

/// Production source backed by the SoX `rec` command.
///
/// Wraps the existing global `RecChild` pool via [`RecChild::get_or_spawn`].
/// Acquires the recorder lazily on first frame read.
#[cfg(feature = "macos")]
pub struct SoxAudioSource {
    handle: std::sync::Arc<Mutex<Option<RecChild>>>,
}

#[cfg(feature = "macos")]
impl SoxAudioSource {
    pub fn new() -> Result<Self> {
        let handle = RecChild::get_or_spawn()?;
        Ok(Self { handle })
    }
}

#[cfg(feature = "macos")]
impl AudioFrameSource for SoxAudioSource {
    fn read_frame(&mut self, buf: &mut [u8; FRAME_BYTES]) -> io::Result<usize> {
        let mut guard = self.handle.lock();
        let Some(rec) = guard.as_mut() else {
            return Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "audio recorder not initialized",
            ));
        };
        match rec.read_frame(buf) {
            // Wedged recorder: evict from the shared slot (Drop kills the
            // child) so the next `get_or_spawn` recovers with a fresh one.
            Err(e) if e.kind() == io::ErrorKind::TimedOut => {
                *guard = None;
                tracing::warn!("rec_timeout_recorder_reset");
                Err(e)
            }
            other => other,
        }
    }
}

/// Linux/Windows stub. Real implementations land with the
/// `linux`/`windows` features in a follow-up.
#[cfg(not(feature = "macos"))]
pub struct StubAudioSource;

#[cfg(not(feature = "macos"))]
impl Default for StubAudioSource {
    fn default() -> Self {
        Self
    }
}

#[cfg(not(feature = "macos"))]
impl AudioFrameSource for StubAudioSource {
    fn read_frame(&mut self, _buf: &mut [u8; FRAME_BYTES]) -> io::Result<usize> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "audio capture is not yet implemented for this platform",
        ))
    }
}

#[cfg(test)]
pub mod mock {
    //! In-memory test double for [`AudioFrameSource`].

    use super::{AudioFrameSource, FRAME_BYTES};
    use std::collections::VecDeque;
    use std::io;

    /// Returns canned frames in FIFO order; once exhausted yields EOF.
    pub struct MockAudioSource {
        frames: VecDeque<[u8; FRAME_BYTES]>,
    }

    impl MockAudioSource {
        pub fn new(frames: Vec<[u8; FRAME_BYTES]>) -> Self {
            Self {
                frames: frames.into(),
            }
        }

        /// Convenience: build a source of `n` silent frames.
        pub fn silence(n: usize) -> Self {
            Self::new(vec![[0u8; FRAME_BYTES]; n])
        }

        pub fn frames_remaining(&self) -> usize {
            self.frames.len()
        }
    }

    impl AudioFrameSource for MockAudioSource {
        fn read_frame(&mut self, buf: &mut [u8; FRAME_BYTES]) -> io::Result<usize> {
            match self.frames.pop_front() {
                Some(frame) => {
                    buf.copy_from_slice(&frame);
                    Ok(FRAME_BYTES)
                }
                None => Ok(0),
            }
        }
    }
}

pub fn temp_wav_path() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let counter = TEMP_WAV_COUNTER.fetch_add(1, Ordering::Relaxed);
    dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("always")
        .join(format!(
            "utterance-{}-{stamp}-{counter}.wav",
            std::process::id()
        ))
}

#[cfg(test)]
mod tests {
    use super::{create_wav_bytes_i16_mono_16k, temp_wav_path};

    #[cfg(feature = "macos")]
    #[test]
    fn wait_readable_times_out_on_silent_pipe() {
        use std::os::fd::AsRawFd as _;
        use std::process::{Command, Stdio};

        // `sleep` holds its stdout pipe open without ever writing — the
        // exact shape of a wedged recorder (no bytes, no EOF).
        let mut child = Command::new("sleep")
            .arg("5")
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn sleep");
        let fd = child.stdout.as_ref().expect("stdout").as_raw_fd();

        let started = std::time::Instant::now();
        let readable = super::wait_readable(fd, 150).expect("poll");
        assert!(!readable, "silent pipe must time out, not read");
        assert!(started.elapsed().as_millis() >= 140);

        let _ = child.kill();
        let _ = child.wait();
    }

    #[cfg(feature = "macos")]
    #[test]
    fn wait_readable_returns_true_when_data_or_eof_arrives() {
        use std::os::fd::AsRawFd as _;
        use std::process::{Command, Stdio};

        let mut child = Command::new("echo")
            .arg("hi")
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn echo");
        let fd = child.stdout.as_ref().expect("stdout").as_raw_fd();

        // Generous timeout — echo writes immediately; poll should return
        // readable long before it.
        assert!(super::wait_readable(fd, 2_000).expect("poll"));

        let _ = child.wait();
    }

    /// A capture helper that dies at start (e.g. VPIO unavailable → exit 4)
    /// must be rejected promptly so `spawn` falls back to `rec`, and must
    /// not be left running (I4: one recorder).
    #[cfg(feature = "macos")]
    #[test]
    fn capture_helper_that_exits_is_rejected_fast() {
        let started = std::time::Instant::now();
        let r = super::RecChild::spawn_capture_helper(std::path::Path::new("/usr/bin/false"));
        assert!(r.is_err(), "an exiting helper must not become the recorder");
        assert!(started.elapsed().as_millis() < 2_000, "EOF must not wait out the timeout");
    }

    /// A helper that streams bytes is accepted and its frames are readable
    /// through the normal `read_frame` path.
    #[cfg(feature = "macos")]
    #[test]
    fn capture_helper_that_streams_is_accepted() {
        let mut rec = super::RecChild::spawn_capture_helper(std::path::Path::new("/usr/bin/yes"))
            .expect("streaming helper accepted");
        let mut buf = [0u8; super::FRAME_BYTES];
        assert_eq!(rec.read_frame(&mut buf).expect("read"), super::FRAME_BYTES);
    }

    /// A helper that never writes is rejected after the first-bytes budget.
    #[cfg(feature = "macos")]
    #[test]
    fn capture_helper_that_stays_silent_is_rejected() {
        let dir = std::env::temp_dir().join(format!("always-cap-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("silent-helper");
        std::fs::write(&script, "#!/bin/sh\nexec sleep 30\n").unwrap();
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let r = super::RecChild::spawn_capture_helper(&script);
        assert!(r.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Live: the real echo-cancelling helper delivers 16 kHz mono s16 at
    /// real-time rate. Needs microphone access; run with --ignored.
    #[cfg(feature = "macos")]
    #[test]
    #[ignore]
    fn live_capture_helper_streams_16k_mono() {
        let Some(path) = super::locate_capture_helper() else {
            panic!("always-capture not built (capture/build.sh)");
        };
        let mut rec = super::RecChild::spawn_capture_helper(&path).expect("helper starts");
        let mut buf = [0u8; super::FRAME_BYTES];
        // Discard start-up burst, then time 100 frames (3.0 s of audio).
        for _ in 0..10 {
            rec.read_frame(&mut buf).unwrap();
        }
        let t = std::time::Instant::now();
        for _ in 0..100 {
            assert_eq!(rec.read_frame(&mut buf).unwrap(), super::FRAME_BYTES);
        }
        let secs = t.elapsed().as_secs_f64();
        eprintln!("100 frames (3.0 s audio) in {secs:.2} s wall");
        assert!((2.6..3.4).contains(&secs), "not real-time 16 kHz: {secs}");
    }

    /// Live: with a broken helper (run with ALWAYS_CAPTURE_BIN=/usr/bin/false)
    /// `spawn` falls back to SoX `rec`, marks the helper failed, and audio
    /// still flows. Needs microphone access; run with --ignored.
    #[cfg(feature = "macos")]
    #[test]
    #[ignore]
    fn live_spawn_falls_back_to_rec_when_helper_fails() {
        if std::env::var("ALWAYS_CAPTURE_BIN").as_deref() != Ok("/usr/bin/false") {
            eprintln!("skipped: set ALWAYS_CAPTURE_BIN=/usr/bin/false");
            return;
        }
        let mut rec = super::RecChild::spawn().expect("fallback recorder");
        assert!(super::CAPTURE_HELPER_FAILED.load(std::sync::atomic::Ordering::Relaxed));
        let mut buf = [0u8; super::FRAME_BYTES];
        assert_eq!(rec.read_frame(&mut buf).unwrap(), super::FRAME_BYTES);
        eprintln!("fallback rec pid streaming");
    }

    #[test]
    fn temp_wav_paths_are_unique() {
        let first = temp_wav_path();
        let second = temp_wav_path();
        assert_ne!(first, second);
    }

    #[test]
    fn wav_bytes_creation_works() {
        let samples = vec![100, -100, 200, -200];
        let wav_data = create_wav_bytes_i16_mono_16k(&samples).unwrap();
        assert!(wav_data.len() > 44); // At least WAV header size
        assert!(wav_data.starts_with(b"RIFF"));
    }

    #[cfg(feature = "macos")]
    #[test]
    fn overrun_window_counts_within_window_and_ages_out() {
        use super::{OverrunWindow, REC_OVERRUN_RESPAWN_THRESHOLD, REC_OVERRUN_WINDOW_SECS};
        use std::sync::atomic::Ordering;

        let w = OverrunWindow::new();
        assert_eq!(w.windowed_count(), 0);
        for i in 1..=REC_OVERRUN_RESPAWN_THRESHOLD {
            assert_eq!(w.record(), i);
        }
        assert_eq!(w.windowed_count(), REC_OVERRUN_RESPAWN_THRESHOLD);

        // Age the window out: a stale window reports 0 (healthy) and the
        // next overrun restarts the count from 1 instead of accumulating
        // forever — the lifetime-counter behavior this replaced.
        w.window_start.store(
            OverrunWindow::now_secs() - REC_OVERRUN_WINDOW_SECS,
            Ordering::Relaxed,
        );
        assert_eq!(w.windowed_count(), 0);
        assert_eq!(w.record(), 1);
        assert_eq!(w.windowed_count(), 1);
    }

    #[test]
    fn audio_buffer_get_drop_get_does_not_leak() {
        use super::AudioBuffer;
        // Pool is process-global so other parallel tests share it; we
        // can't assert capacity reuse deterministically. Assert the basic
        // contract: get + drop + get does not panic and produces usable
        // buffers.
        let cap_before;
        {
            let mut buf = AudioBuffer::get();
            buf.as_mut().reserve(32_768);
            cap_before = buf.as_mut().capacity();
        }
        let _buf = AudioBuffer::get();
        assert!(cap_before >= 32_768);
    }

    #[test]
    fn audio_buffer_pool_caps_at_ten_entries() {
        use super::AudioBuffer;
        // Deluge the pool with 20 returns. The pool length is capped at 10
        // by Drop logic. We can only observe via the absence of OOM and
        // that subsequent gets succeed.
        let mut bufs = Vec::new();
        for _ in 0..20 {
            bufs.push(AudioBuffer::get());
        }
        drop(bufs);
        // Confirm a fresh get still works after the deluge.
        let _ = AudioBuffer::get();
    }

    #[test]
    fn mock_audio_source_yields_frames_then_eof() {
        use super::AudioFrameSource;
        use super::FRAME_BYTES;
        use super::mock::MockAudioSource;

        let mut src = MockAudioSource::silence(2);
        let mut buf = [0u8; FRAME_BYTES];

        assert_eq!(src.read_frame(&mut buf).unwrap(), FRAME_BYTES);
        assert_eq!(src.read_frame(&mut buf).unwrap(), FRAME_BYTES);
        // After exhaustion: EOF (Ok(0)).
        assert_eq!(src.read_frame(&mut buf).unwrap(), 0);
    }

    #[test]
    fn mock_audio_source_preserves_frame_contents() {
        use super::AudioFrameSource;
        use super::FRAME_BYTES;
        use super::mock::MockAudioSource;

        let mut frame_a = [0u8; FRAME_BYTES];
        frame_a[0] = 0xAB;
        frame_a[1] = 0xCD;
        let mut src = MockAudioSource::new(vec![frame_a]);

        let mut buf = [0u8; FRAME_BYTES];
        assert_eq!(src.read_frame(&mut buf).unwrap(), FRAME_BYTES);
        assert_eq!(buf[0], 0xAB);
        assert_eq!(buf[1], 0xCD);
    }
}
