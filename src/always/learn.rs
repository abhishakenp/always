//! Learning ring: HIS OWN accepted voice, kept on this Mac so the voice
//! models keep learning from him.
//!
//! The enrollment (`voiceprint.json`, a 256-dim voiceprint, no audio) is
//! the base. On top of it Always keeps, locally and bounded, the
//! utterances he actually dictated: My Voice verified them (speaker gate
//! passed), the endpoint ended them, and they were pasted. With them it
//! keeps one feature row per pause (the `eot` prosody features at each
//! probe frame, labelled `resumed` = he went on, or END). The end-of-turn
//! model is retrained from these rows while he is away (`spawn_retrainer`)
//! and promoted only when its replay wins.
//!
//! Consent / privacy: on by default for the owner (he consented,
//! 2026-10-16); `always config set learn_from_voice false` stops all
//! collection and training. Nothing here is ever uploaded. Gate off →
//! nothing is kept (the voice could be anyone's).
//!
//! ## On-disk format (stable; other components read it)
//!
//! `<config_dir>/learn/` = `~/Library/Application Support/always/learn/`
//! (`always-dev/learn/` for the dev instance; `ALWAYS_LEARN_DIR`
//! overrides). Directories are 0700, files 0600.
//!
//! - `utt/<id>.wav` — 16 kHz mono s16le WAV of one accepted utterance
//!   (after My Voice edge trimming). A chunked dictation keeps only its
//!   live tail (`tail_only`), at most `MAX_UTT_SECS`.
//! - `utts.jsonl` — one [`UttMeta`] per line, oldest first. The ring holds
//!   at most [`AUDIO_CAP_SECS`] of audio; the oldest utterances (wav +
//!   meta line) are dropped first.
//! - `pauses.jsonl` — one [`crate::always::eot::PauseRow`] per pause
//!   (`file` = utterance id), at most [`ROWS_CAP`] rows, oldest dropped.
//! - `relabels.jsonl` — `{"file": <id>, "gap_ms": n}`: he spoke again
//!   `gap_ms` after that utterance was cut, so its END was really a
//!   thinking pause (a split). Applied by [`load_rows`].
//! - `seed-rows.jsonl` (optional) — corpus rows from his older audio.
//! - `models/eot-v<N>.json` + `models/state.json` — versioned end-of-turn
//!   models; `state.json` = `{"current": N|null, "last_good": M|null}`.
//!
//! ## Rust interface (for other consumers, e.g. a speaker extractor)
//!
//! - [`dir`], [`enabled`]
//! - [`list`] → `Vec<UttMeta>` (oldest first), [`load_audio`]`(id)` →
//!   samples, [`total_audio_secs`]
//! - [`load_rows`] → labelled pause rows (seed + ring + relabels)
//! - [`eot_model`] → the promoted end-of-turn model, if any
//!
//! Hot-path cost: [`stage`] copies the utterance into a pooled buffer
//! (no allocation once warm) and [`commit_staged`] hands it to a
//! background-QoS writer thread with `try_send` — the capture/paste path
//! never waits on disk; if the writer is behind, the utterance is skipped.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::always::eot::{EotModel, PauseRow};

/// Audio kept in the ring (30 minutes).
pub const AUDIO_CAP_SECS: f64 = 30.0 * 60.0;
/// Pause rows kept.
pub const ROWS_CAP: usize = 10_000;
/// Audio kept per utterance (its last 2 minutes).
pub const MAX_UTT_SECS: usize = 120;
/// He spoke again within this of a cut: that END was a split.
pub const RELABEL_GAP_MS: u64 = 2_000;

static ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// Collection + training on (`learn_from_voice` preference).
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn dir() -> PathBuf {
    if let Ok(p) = std::env::var("ALWAYS_LEARN_DIR") {
        return PathBuf::from(p);
    }
    crate::config::config_dir().join("learn")
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct UttMeta {
    /// Unique id (`<unix_ms of the cut>`), also the wav file stem.
    pub id: String,
    /// Wall clock of the endpoint cut.
    pub cut_unix_ms: u64,
    /// Wall clock of his first voiced frame.
    pub voice_start_unix_ms: u64,
    /// Seconds of audio in the wav.
    pub secs: f32,
    /// Chunked dictation: the wav is only the live tail.
    pub tail_only: bool,
    /// Ended by the learned early cut (vs the base window).
    pub early_cut: bool,
    /// Raw silence (ms) from his last voiced frame to the cut.
    pub raw_silence_ms: u32,
    pub pauses: u32,
}

struct Staged {
    meta: UttMeta,
    samples: Vec<i16>,
    rows: Vec<PauseRow>,
}

static STAGED: Mutex<Option<Staged>> = Mutex::new(None);
/// Reusable sample buffers (returned by the writer after each write).
static POOL: Mutex<Vec<Vec<i16>>> = Mutex::new(Vec::new());
const POOL_BUF_SAMPLES: usize = 30 * 16_000;

fn lock<T>(m: &'static Mutex<T>) -> std::sync::MutexGuard<'static, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Capture thread, after My Voice accepted the utterance: keep its audio
/// and pause rows until the event loop knows it was pasted. `rows` is
/// taken (left empty). Replaces any earlier staged utterance.
pub fn stage(samples: &[i16], rows: &mut Vec<PauseRow>, meta: UttMeta) {
    if !enabled() {
        return;
    }
    let keep = &samples[samples.len().saturating_sub(MAX_UTT_SECS * 16_000)..];
    let mut buf = lock(&POOL).pop().unwrap_or_default();
    buf.clear();
    buf.extend_from_slice(keep);
    let old = lock(&STAGED).replace(Staged {
        meta,
        samples: buf,
        rows: std::mem::take(rows),
    });
    if let Some(old) = old {
        recycle(old.samples);
    }
}

/// The staged utterance was pasted: hand it to the writer (never blocks).
pub fn commit_staged() {
    let Some(s) = lock(&STAGED).take() else {
        return;
    };
    if !enabled() {
        recycle(s.samples);
        return;
    }
    match writer().try_send(s) {
        Ok(()) => {}
        Err(mpsc::TrySendError::Full(s)) | Err(mpsc::TrySendError::Disconnected(s)) => {
            tracing::info!("learn_ring_writer_busy_skipped");
            recycle(s.samples);
        }
    }
}

/// Not pasted (filtered, consumed, dropped): forget it.
pub fn discard_staged() {
    if let Some(s) = lock(&STAGED).take() {
        recycle(s.samples);
    }
}

fn recycle(mut buf: Vec<i16>) {
    if buf.capacity() > POOL_BUF_SAMPLES * 5 {
        return; // a rare 2-minute buffer is not kept around
    }
    buf.clear();
    let mut pool = lock(&POOL);
    if pool.len() < 2 {
        pool.push(buf);
    }
}

fn writer() -> &'static mpsc::SyncSender<Staged> {
    static TX: std::sync::OnceLock<mpsc::SyncSender<Staged>> = std::sync::OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = mpsc::sync_channel::<Staged>(2);
        // Pre-fill the pool so the first stage does not allocate on the
        // capture thread either.
        {
            let mut pool = lock(&POOL);
            while pool.len() < 2 {
                pool.push(Vec::with_capacity(POOL_BUF_SAMPLES));
            }
        }
        std::thread::Builder::new()
            .name("learn-ring".into())
            .spawn(move || {
                background_qos();
                let mut ring = Ring::open(dir());
                for s in rx {
                    if let Err(e) = ring.write(&s.meta, &s.samples, &s.rows) {
                        tracing::warn!(error = %e, "learn_ring_write_failed");
                    }
                    recycle(s.samples);
                }
            })
            .expect("spawn learn-ring thread");
        tx
    })
}

/// Pre-warm the writer + pool at daemon start (off the capture path).
pub fn init(on: bool) {
    set_enabled(on);
    if on {
        let _ = writer();
    }
}

/// Lowest scheduling class for this thread (Darwin background QoS: low
/// CPU priority, throttled I/O, efficiency cores where present).
pub(crate) fn background_qos() {
    #[cfg(target_os = "macos")]
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_BACKGROUND, 0);
    }
}

// ---------------------------------------------------------------------
// The ring on disk.

fn create_private_dir(d: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(d)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(d, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn open_private(path: &Path, append: bool) -> std::io::Result<std::fs::File> {
    let mut o = std::fs::OpenOptions::new();
    o.create(true);
    if append {
        o.append(true);
    } else {
        o.write(true).truncate(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    let f = o.open(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        f.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(f)
}

/// Atomic private rewrite (tmp + rename).
fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    let mut f = open_private(&tmp, false)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    std::fs::rename(&tmp, path)
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Vec<T> {
    std::fs::read_to_string(path)
        .map(|t| t.lines().filter_map(|l| serde_json::from_str(l).ok()).collect())
        .unwrap_or_default()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Relabel {
    file: String,
    gap_ms: u64,
}

struct Ring {
    dir: PathBuf,
    metas: Vec<UttMeta>,
    rows: usize,
    last: Option<UttMeta>,
}

impl Ring {
    fn open(dir: PathBuf) -> Self {
        let metas: Vec<UttMeta> = read_jsonl(&dir.join("utts.jsonl"));
        let rows = std::fs::read_to_string(dir.join("pauses.jsonl"))
            .map(|t| t.lines().count())
            .unwrap_or(0);
        let last = metas.last().cloned();
        Self {
            dir,
            metas,
            rows,
            last,
        }
    }

    fn write(&mut self, meta: &UttMeta, samples: &[i16], rows: &[PauseRow]) -> anyhow::Result<()> {
        create_private_dir(&self.dir)?;
        create_private_dir(&self.dir.join("utt"))?;
        let wav = crate::always::audio::create_wav_bytes_i16_mono_16k(samples)?;
        write_private(&self.dir.join("utt").join(format!("{}.wav", meta.id)), &wav)?;
        let mut line = serde_json::to_string(meta)?;
        line.push('\n');
        open_private(&self.dir.join("utts.jsonl"), true)?.write_all(line.as_bytes())?;
        // Pause rows: global sequence = cut time (ms) * 1000 + index.
        let mut out = String::new();
        for (i, r) in rows.iter().enumerate() {
            let mut r = r.clone();
            r.file = meta.id.clone();
            r.seq = meta.cut_unix_ms * 1000 + i as u64 + 1;
            out.push_str(&serde_json::to_string(&r)?);
            out.push('\n');
        }
        if !out.is_empty() {
            open_private(&self.dir.join("pauses.jsonl"), true)?.write_all(out.as_bytes())?;
            self.rows += rows.len();
            NEW_ROWS.fetch_add(rows.len() as u64, Ordering::Relaxed);
        }
        // He resumed right after the previous cut: that END was a split.
        if let Some(prev) = &self.last {
            let gap = meta.voice_start_unix_ms.saturating_sub(prev.cut_unix_ms);
            if meta.voice_start_unix_ms >= prev.cut_unix_ms && gap < RELABEL_GAP_MS {
                let mut l = serde_json::to_string(&Relabel {
                    file: prev.id.clone(),
                    gap_ms: gap,
                })?;
                l.push('\n');
                open_private(&self.dir.join("relabels.jsonl"), true)?.write_all(l.as_bytes())?;
                NEW_ROWS.fetch_add(1, Ordering::Relaxed);
                tracing::info!(gap_ms = gap, early = prev.early_cut, "learn_end_relabelled_split");
            }
        }
        self.metas.push(meta.clone());
        self.last = Some(meta.clone());
        self.prune()?;
        Ok(())
    }

    fn prune(&mut self) -> anyhow::Result<()> {
        let total: f64 = self.metas.iter().map(|m| m.secs as f64).sum();
        if total > AUDIO_CAP_SECS {
            let mut excess = total - AUDIO_CAP_SECS;
            let mut drop = 0;
            for m in &self.metas {
                if excess <= 0.0 {
                    break;
                }
                excess -= m.secs as f64;
                let _ = std::fs::remove_file(self.dir.join("utt").join(format!("{}.wav", m.id)));
                drop += 1;
            }
            self.metas.drain(..drop);
            let mut text = String::new();
            for m in &self.metas {
                text.push_str(&serde_json::to_string(m)?);
                text.push('\n');
            }
            write_private(&self.dir.join("utts.jsonl"), text.as_bytes())?;
        }
        // Rows: rewrite to the newest ROWS_CAP once 25 % over (amortised).
        if self.rows > ROWS_CAP + ROWS_CAP / 4 {
            let p = self.dir.join("pauses.jsonl");
            let text = std::fs::read_to_string(&p)?;
            let lines: Vec<&str> = text.lines().collect();
            let keep = &lines[lines.len().saturating_sub(ROWS_CAP)..];
            let mut out = keep.join("\n");
            out.push('\n');
            write_private(&p, out.as_bytes())?;
            self.rows = keep.len();
        }
        Ok(())
    }
}

/// Pause rows written since the last training attempt (+ relabels).
pub(crate) static NEW_ROWS: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------
// Reading API.

/// Accepted utterances in the ring, oldest first.
pub fn list() -> Vec<UttMeta> {
    read_jsonl(&dir().join("utts.jsonl"))
}

pub fn total_audio_secs() -> f64 {
    list().iter().map(|m| m.secs as f64).sum()
}

/// Samples of one ring utterance (16 kHz mono).
pub fn load_audio(id: &str) -> Option<Vec<i16>> {
    if id.contains('/') || id.contains("..") {
        return None;
    }
    let b = std::fs::read(dir().join("utt").join(format!("{id}.wav"))).ok()?;
    let data = b.get(44..)?;
    Some(
        data.chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect(),
    )
}

/// Labelled pause rows: seed corpus + ring, with relabels applied (an
/// END he resumed within `RELABEL_GAP_MS` of becomes a continuation).
pub fn load_rows() -> Vec<PauseRow> {
    load_rows_in(&dir())
}

pub(crate) fn load_rows_in(d: &Path) -> Vec<PauseRow> {
    let mut rows: Vec<PauseRow> = read_jsonl(&d.join("seed-rows.jsonl"));
    let mut ring: Vec<PauseRow> = read_jsonl(&d.join("pauses.jsonl"));
    let relabels: Vec<Relabel> = read_jsonl(&d.join("relabels.jsonl"));
    for rl in &relabels {
        if let Some(r) = ring.iter_mut().rev().find(|r| r.file == rl.file && !r.resumed) {
            r.resumed = true;
            r.raw_len += (rl.gap_ms / 30) as u32;
        }
    }
    rows.append(&mut ring);
    rows
}

// ---------------------------------------------------------------------
// Versioned end-of-turn models.

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelState {
    pub current: Option<u32>,
    pub last_good: Option<u32>,
}

fn models_dir(d: &Path) -> PathBuf {
    d.join("models")
}

fn model_file(d: &Path, v: u32) -> PathBuf {
    models_dir(d).join(format!("eot-v{v}.json"))
}

pub(crate) fn read_state(d: &Path) -> ModelState {
    std::fs::read_to_string(models_dir(d).join("state.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

pub(crate) fn read_model(d: &Path, v: u32) -> Option<EotModel> {
    let t = std::fs::read_to_string(model_file(d, v)).ok()?;
    let m: EotModel = serde_json::from_str(&t).ok()?;
    m.is_valid().then_some(m)
}

fn write_state(d: &Path, s: &ModelState) -> anyhow::Result<()> {
    create_private_dir(&models_dir(d))?;
    write_private(&models_dir(d).join("state.json"), serde_json::to_string_pretty(s)?.as_bytes())?;
    Ok(())
}

/// Store `m` as the next version and make it current; the previous
/// current becomes `last_good` (rollback target). Returns the version.
pub(crate) fn promote_in(d: &Path, mut m: EotModel) -> anyhow::Result<u32> {
    create_private_dir(d)?;
    create_private_dir(&models_dir(d))?;
    let mut st = read_state(d);
    let next = std::fs::read_dir(models_dir(d))
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| {
                    e.file_name()
                        .to_str()?
                        .strip_prefix("eot-v")?
                        .strip_suffix(".json")?
                        .parse::<u32>()
                        .ok()
                })
                .max()
                .unwrap_or(0)
        })
        .unwrap_or(0)
        + 1;
    m.version = next;
    write_private(&model_file(d, next), serde_json::to_string_pretty(&m)?.as_bytes())?;
    if st.current.is_some() {
        st.last_good = st.current;
    }
    st.current = Some(next);
    write_state(d, &st)?;
    Ok(next)
}

/// Current model is splitting his thoughts: fall back to `last_good`
/// (or to no early cut at all when `to_last_good` is false / missing).
pub(crate) fn rollback_in(d: &Path, to_last_good: bool) -> anyhow::Result<Option<u32>> {
    let mut st = read_state(d);
    st.current = if to_last_good { st.last_good } else { None };
    st.last_good = None;
    write_state(d, &st)?;
    Ok(st.current)
}

static MODEL: RwLock<Option<Option<Arc<EotModel>>>> = RwLock::new(None);

/// The promoted end-of-turn model (`None` = no early cut). Cached; the
/// retrainer refreshes it on promote / rollback.
pub fn eot_model() -> Option<Arc<EotModel>> {
    if let Some(m) = MODEL.read().as_ref() {
        return m.clone();
    }
    let d = dir();
    let m = read_state(&d).current.and_then(|v| read_model(&d, v)).map(Arc::new);
    *MODEL.write() = Some(m.clone());
    m
}

pub(crate) fn reload_model() {
    *MODEL.write() = None;
    let m = eot_model();
    tracing::info!(version = m.as_ref().map(|m| m.version), "eot_model_loaded");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::always::eot::NF;

    fn tmp_dir(name: &str) -> PathBuf {
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/test-tmp")
            .join(format!("learn-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn row(end: bool, raw: u32) -> PauseRow {
        PauseRow {
            raw_len: raw,
            resumed: !end,
            probes: vec![(5, [0.5; NF])],
            ..PauseRow::default()
        }
    }

    fn meta(id: u64, start: u64, secs: f32) -> UttMeta {
        UttMeta {
            id: id.to_string(),
            cut_unix_ms: id,
            voice_start_unix_ms: start,
            secs,
            ..UttMeta::default()
        }
    }

    #[test]
    fn ring_writes_private_files_bounds_audio_and_relabels_splits() {
        let d = tmp_dir("ring");
        let mut ring = Ring::open(d.clone());
        let one_sec = vec![100i16; 16_000];
        ring.write(&meta(10_000, 5_000, 1.0), &one_sec, &[row(false, 12), row(true, 9)])
            .unwrap();
        // Resumed 800 ms after the cut: the previous END becomes a split.
        ring.write(&meta(20_000, 10_800, 1.0), &one_sec, &[row(true, 30)]).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |p: PathBuf| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(d.clone()), 0o700);
            assert_eq!(mode(d.join("utt").join("10000.wav")), 0o600);
            assert_eq!(mode(d.join("utts.jsonl")), 0o600);
            assert_eq!(mode(d.join("pauses.jsonl")), 0o600);
            assert_eq!(mode(d.join("relabels.jsonl")), 0o600);
        }
        let rows = load_rows_in(&d);
        assert_eq!(rows.len(), 3);
        assert!(rows[0].resumed && rows[0].seq == 10_000_001);
        assert!(rows[1].resumed, "relabelled END: {:?}", rows[1]);
        assert_eq!(rows[1].raw_len, 9 + 800 / 30);
        assert!(!rows[2].resumed && rows[2].file == "20000");
        // Audio cap: oldest dropped first (wav + meta).
        let big = vec![1i16; 16_000 * 10];
        let mut id = 30_000u64;
        let n = (AUDIO_CAP_SECS / 10.0) as usize + 2;
        for _ in 0..n {
            ring.write(&meta(id, id - 9_000, 10.0), &big, &[]).unwrap();
            id += 100_000;
        }
        let total: f64 = ring.metas.iter().map(|m| m.secs as f64).sum();
        assert!(total <= AUDIO_CAP_SECS, "{total}");
        assert!(!d.join("utt").join("10000.wav").exists());
        assert_eq!(read_jsonl::<UttMeta>(&d.join("utts.jsonl")).len(), ring.metas.len());
        let wavs = std::fs::read_dir(d.join("utt")).unwrap().count();
        assert_eq!(wavs, ring.metas.len());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn models_are_versioned_with_rollback() {
        let d = tmp_dir("models");
        let m = |b: f32| EotModel {
            probes: vec![crate::always::eot::EotProbe {
                probe_frames: 8,
                mean: vec![0.0; NF],
                std: vec![1.0; NF],
                w: vec![0.1; NF],
                b,
                threshold: 0.9,
            }],
            ..EotModel::default()
        };
        assert_eq!(promote_in(&d, m(1.0)).unwrap(), 1);
        assert_eq!(promote_in(&d, m(2.0)).unwrap(), 2);
        let st = read_state(&d);
        assert_eq!((st.current, st.last_good), (Some(2), Some(1)));
        assert_eq!(read_model(&d, 2).unwrap().version, 2);
        assert_eq!(rollback_in(&d, true).unwrap(), Some(1));
        assert_eq!(read_state(&d).current, Some(1));
        assert_eq!(promote_in(&d, m(3.0)).unwrap(), 3, "versions never reused");
        assert_eq!(rollback_in(&d, false).unwrap(), None);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Hot-path cost of staging a 10 s utterance (pooled copy, no
    /// allocation once warm); off → nothing is kept. One test: both halves
    /// toggle the process-wide switch.
    #[test]
    fn stage_is_cheap_and_disabled_keeps_nothing() {
        set_enabled(false);
        let mut rows = vec![row(true, 9)];
        stage(&[1i16; 1000], &mut rows, UttMeta::default());
        assert!(lock(&STAGED).is_none(), "off: nothing staged");
        assert_eq!(rows.len(), 1, "off: rows untouched");

        set_enabled(true);
        {
            let mut pool = lock(&POOL);
            while pool.len() < 2 {
                pool.push(Vec::with_capacity(POOL_BUF_SAMPLES));
            }
        }
        let audio = vec![7i16; 16_000 * 10];
        let mut worst = std::time::Duration::ZERO;
        for _ in 0..50 {
            let mut rows = vec![row(true, 9), row(false, 12)];
            let t = std::time::Instant::now();
            stage(&audio, &mut rows, UttMeta::default());
            worst = worst.max(t.elapsed());
            assert!(rows.is_empty(), "rows taken");
            discard_staged();
        }
        set_enabled(false);
        eprintln!("learn stage 10 s utterance: worst {} µs", worst.as_micros());
        assert!(worst < std::time::Duration::from_millis(5), "{worst:?}");
    }
}
