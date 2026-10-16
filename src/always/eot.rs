//! Learned, per-user end-of-turn: cut early ONLY when his own voice says
//! he is done, otherwise leave the endpoint exactly as it is.
//!
//! The endpoint's silence count starts after a 10-frame (300 ms) max-hold
//! of Silero's probability (`smoothing_window` in `vad.rs`), so the base
//! 0.6 s window really ends ~0.9 s after his last voiced frame. This
//! module watches the RAW frames (no smoothing) and, at fixed probe
//! points of raw silence, scores the pause with a tiny logistic model
//! over prosody (final energy / pitch movement, final-run lengthening,
//! breath in the pause, speaking rate). Only a score above a threshold
//! chosen on his own data above every continuation pause's score ends the
//! utterance early; anything else falls through to the unchanged
//! endpoint. No model file → no early cut, ever.
//!
//! Cost: one autocorrelation pitch track per voiced 30 ms frame
//! (decimated to 8 kHz, ~34 k MACs) and a dot product of `NF` floats
//! per probe; state is a fixed 100-frame ring + two small histograms.

use std::collections::VecDeque;

/// One frame (ms).
const FRAME_MS: u32 = 30;
/// Frames of history kept (3 s at 30 ms).
const HISTORY: usize = 100;
/// Raw Silero probability counted as voiced (the endpoint's in-speech
/// `silence_threshold` at the default 0.5 Silero threshold).
pub const VOICED_PROB: f32 = 0.375;
/// A gap this long (frames) ends a voiced run (syllable-ish unit).
const RUN_GAP: u32 = 2;
/// Pauses (raw silence runs) at least this long count for the pause rate.
const PAUSE_MIN: u32 = 5;
/// Number of features.
pub const NF: usize = 11;
pub const FEATURE_NAMES: [&str; NF] = [
    "final_db_rel",
    "db_slope",
    "f0_slope",
    "final_f0_rel",
    "final_run_rel",
    "ln_utt_secs",
    "run_rate",
    "pause_rate",
    "breath_db_rel",
    "sil_prob_mean",
    "final_voicing",
];

#[derive(Clone, Copy, Default, Debug)]
struct Obs {
    prob: f32,
    db: f32,
    /// Semitones re 55 Hz; 0 = no pitch.
    st: f32,
    voiced: bool,
}

/// Per-utterance raw-frame tracker. `reset` at each utterance start,
/// `push` every frame while in speech (including the pause frames).
#[derive(Clone, Debug)]
pub struct EotTracker {
    ring: VecDeque<Obs>,
    raw_silence: u32,
    frames: u32,
    db_hist: [u32; 64],
    st_hist: [u32; 64],
    runs: u32,
    run_len: u32,
    last_run_len: u32,
    run_len_sum: u32,
    run_count: u32,
    pauses: u32,
    seen_voice: bool,
    /// This pause must not be cut early (Option held, own cue echo).
    held: bool,
}

impl Default for EotTracker {
    fn default() -> Self {
        Self {
            ring: VecDeque::with_capacity(HISTORY),
            raw_silence: 0,
            frames: 0,
            db_hist: [0; 64],
            st_hist: [0; 64],
            runs: 0,
            run_len: 0,
            last_run_len: 0,
            run_len_sum: 0,
            run_count: 0,
            pauses: 0,
            seen_voice: false,
            held: false,
        }
    }
}

fn frame_db(samples: &[i16]) -> f32 {
    let ss: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    let rms = (ss / samples.len().max(1) as f64).sqrt() / 32768.0;
    (20.0 * (rms + 1e-6).log10()) as f32
}

/// Autocorrelation pitch on a 30 ms frame decimated to 8 kHz; semitones
/// re 55 Hz, or `None` when not clearly periodic (70-400 Hz).
fn pitch_st(samples: &[i16]) -> Option<f32> {
    let n = samples.len() / 2;
    let mut x = [0f32; 240];
    let n = n.min(240);
    let mut mean = 0f32;
    for (i, slot) in x.iter_mut().enumerate().take(n) {
        *slot = (samples[2 * i] as f32 + samples[2 * i + 1] as f32) * 0.5;
        mean += *slot;
    }
    mean /= n as f32;
    let mut e0 = 0f32;
    for v in x.iter_mut().take(n) {
        *v -= mean;
        e0 += *v * *v;
    }
    if e0 <= 0.0 {
        return None;
    }
    let (lo, hi) = (20usize, 114usize); // 400 Hz .. 70 Hz at 8 kHz
    let mut best = (0usize, 0f32);
    for lag in lo..=hi.min(n - 1) {
        let mut c = 0f32;
        let mut e1 = 0f32;
        let mut e2 = 0f32;
        for i in 0..n - lag {
            c += x[i] * x[i + lag];
            e1 += x[i] * x[i];
            e2 += x[i + lag] * x[i + lag];
        }
        let r = c / (e1 * e2).sqrt().max(1e-9);
        if r > best.1 {
            best = (lag, r);
        }
    }
    (best.1 >= 0.55).then(|| 12.0 * (8000.0 / best.0 as f32 / 55.0).log2())
}

fn hist_median(h: &[u32; 64]) -> Option<f32> {
    let total: u32 = h.iter().sum();
    if total == 0 {
        return None;
    }
    let mut acc = 0;
    for (i, &c) in h.iter().enumerate() {
        acc += c;
        if acc * 2 >= total {
            return Some(i as f32);
        }
    }
    None
}

/// Least-squares slope of `ys` per index step.
fn slope(ys: &[f32]) -> f32 {
    let n = ys.len() as f32;
    if ys.len() < 3 {
        return 0.0;
    }
    let mx = (n - 1.0) / 2.0;
    let my = ys.iter().sum::<f32>() / n;
    let (mut num, mut den) = (0f32, 0f32);
    for (i, y) in ys.iter().enumerate() {
        let dx = i as f32 - mx;
        num += dx * (y - my);
        den += dx * dx;
    }
    num / den.max(1e-9)
}

impl EotTracker {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Frames of raw silence since his last voiced frame.
    pub fn raw_silence_frames(&self) -> u32 {
        self.raw_silence
    }

    pub fn seen_voice(&self) -> bool {
        self.seen_voice
    }

    /// Never cut the current pause early (cleared by his next voiced frame).
    pub fn hold(&mut self) {
        self.held = true;
    }

    pub fn held(&self) -> bool {
        self.held
    }

    /// One 30 ms frame. `voiced` = the raw (unsmoothed) speech evidence:
    /// Silero prob ≥ `VOICED_PROB` or energy above the in-speech floor.
    pub fn push(&mut self, samples: &[i16], prob: f32, voiced: bool) {
        let db = frame_db(samples);
        let st = if voiced && prob >= 0.5 {
            pitch_st(samples).unwrap_or(0.0)
        } else {
            0.0
        };
        if self.ring.len() == HISTORY {
            self.ring.pop_front();
        }
        self.ring.push_back(Obs {
            prob,
            db,
            st,
            voiced,
        });
        self.frames += 1;
        if voiced {
            if self.raw_silence >= PAUSE_MIN && self.seen_voice {
                self.pauses += 1;
            }
            if self.raw_silence >= RUN_GAP || self.run_len == 0 {
                self.runs += 1;
                self.run_len = 0;
            }
            self.run_len += 1;
            self.raw_silence = 0;
            self.seen_voice = true;
            self.held = false;
            let b = ((db + 80.0) / 1.25).clamp(0.0, 63.0) as usize;
            self.db_hist[b] += 1;
            if st > 0.0 {
                self.st_hist[(st.clamp(0.0, 63.0)) as usize] += 1;
            }
        } else {
            self.raw_silence += 1;
            if self.raw_silence == RUN_GAP && self.run_len > 0 {
                self.last_run_len = self.run_len;
                self.run_len_sum += self.run_len;
                self.run_count += 1;
                self.run_len = 0;
            }
        }
    }

    /// Feature vector at the current pause (call only while
    /// `raw_silence_frames() > 0`). Allocation-free: it runs at every
    /// probe frame of every pause on the capture thread.
    pub fn features(&self) -> [f32; NF] {
        let n = self.ring.len();
        let sil = (self.raw_silence as usize).min(n);
        let speech_end = n - sil;
        // Last 8 voiced frames before the pause, oldest first.
        let mut tail_buf = [Obs::default(); 8];
        let mut tn = 0usize;
        for o in self.ring.iter().take(speech_end).rev() {
            if o.voiced {
                tail_buf[7 - tn] = *o;
                tn += 1;
                if tn == 8 {
                    break;
                }
            }
        }
        let tail = &tail_buf[8 - tn..];
        let last3 = &tail[tn.saturating_sub(3)..];
        let med_db = hist_median(&self.db_hist).map_or(-30.0, |b| b * 1.25 - 80.0);
        let med_st = hist_median(&self.st_hist);
        let final_db = if last3.is_empty() {
            med_db
        } else {
            last3.iter().map(|o| o.db).sum::<f32>() / last3.len() as f32
        };
        let mut dbs = [0f32; 8];
        for (d, o) in dbs.iter_mut().zip(tail) {
            *d = o.db;
        }
        let mut sts = [0f32; 8];
        let mut sn = 0usize;
        for o in tail.iter().filter(|o| o.st > 0.0) {
            sts[sn] = o.st;
            sn += 1;
        }
        let (mut st3_sum, mut st3_n) = (0f32, 0usize);
        for o in last3.iter().filter(|o| o.st > 0.0) {
            st3_sum += o.st;
            st3_n += 1;
        }
        let final_st_rel = match (med_st, st3_n == 0) {
            (Some(m), false) => st3_sum / st3_n as f32 - (m + 0.5),
            _ => 0.0,
        };
        let mean_run = if self.run_count > 0 {
            self.run_len_sum as f32 / self.run_count as f32
        } else {
            1.0
        };
        let secs = (self.frames as f32 * 0.03).max(0.3);
        let (mut breath, mut prob_sum) = (f32::MIN, 0f32);
        for o in self.ring.iter().skip(speech_end) {
            breath = breath.max(o.db);
            prob_sum += o.prob;
        }
        let breath = breath.max(-100.0);
        let sil_prob = if sil == 0 { 0.0 } else { prob_sum / sil as f32 };
        let voicing = if tail.is_empty() {
            0.0
        } else {
            sn as f32 / tail.len() as f32
        };
        [
            final_db - med_db,
            slope(&dbs[..tn]),
            if sn >= 3 { slope(&sts[..sn]) } else { 0.0 },
            final_st_rel,
            self.last_run_len as f32 / mean_run.max(1.0),
            secs.ln(),
            self.runs as f32 / secs,
            self.pauses as f32 / secs,
            breath - med_db,
            sil_prob,
            voicing,
        ]
    }
}

/// One probe of the model: at `probe_frames` of raw silence, cut when
/// the standardized logistic score is at least `threshold`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EotProbe {
    pub probe_frames: u32,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub w: Vec<f32>,
    pub b: f32,
    pub threshold: f32,
}

impl EotProbe {
    pub fn score(&self, f: &[f32; NF]) -> f32 {
        let mut z = self.b;
        for i in 0..NF {
            z += self.w[i] * (f[i] - self.mean[i]) / self.std[i].max(1e-6);
        }
        1.0 / (1.0 + (-z).exp())
    }
}

/// The per-user model: probes in increasing order of raw silence.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct EotModel {
    pub probes: Vec<EotProbe>,
    /// Free-form provenance (data size, validation numbers).
    #[serde(default)]
    pub note: String,
    /// Monotonic version under `learn/models/` (0 = unversioned).
    #[serde(default)]
    pub version: u32,
    /// Pause rows the model was trained on (the newest row's sequence
    /// number), so later rows are true held-out data for it.
    #[serde(default)]
    pub trained_through_seq: u64,
}

impl EotModel {
    /// Valid shape (feature count, ordering) — a malformed file is ignored.
    pub fn is_valid(&self) -> bool {
        !self.probes.is_empty()
            && self.probes.windows(2).all(|w| w[0].probe_frames < w[1].probe_frames)
            && self.probes.iter().all(|p| {
                p.w.len() == NF
                    && p.mean.len() == NF
                    && p.std.len() == NF
                    && p.threshold.is_finite()
                    && p.w.iter().chain(&p.mean).chain(&p.std).all(|v| v.is_finite())
            })
    }

    /// At exactly a probe frame: does the model say he is done?
    pub fn cut_now(&self, tracker: &EotTracker) -> Option<f32> {
        if tracker.held() || !tracker.seen_voice() {
            return None;
        }
        let s = tracker.raw_silence_frames();
        let p = self.probes.iter().find(|p| p.probe_frames == s)?;
        let score = p.score(&tracker.features());
        (score > p.threshold).then_some(score)
    }

    /// Offline twin of `cut_now` over a recorded pause: the first probe
    /// (in increasing silence) whose score clears its threshold.
    pub fn first_cut(&self, row: &PauseRow) -> Option<u32> {
        for p in &self.probes {
            if let Some((_, f)) = row.probes.iter().find(|(k, _)| *k == p.probe_frames)
                && p.score(f) > p.threshold
            {
                return Some(p.probe_frames);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------
// Pause rows: one per raw pause (≥ PROBE_MIN frames) of an utterance —
// the same schema for the offline corpus (`eot_corpus`) and the live
// learning ring (`learn`).

/// First / last raw-silence frame at which features are recorded.
pub const PROBE_MIN: u32 = 3;
pub const PROBE_MAX: u32 = 24;
/// Base endpoint window (0.6 s) in frames, and the max-hold of Silero's
/// probability in `vad.rs` (`smoothing_window`): the base cut lands
/// ~`SMOOTHING_HOLD + BASE_FRAMES - 1` raw frames after his last word.
pub const BASE_FRAMES: u32 = 20;
pub const SMOOTHING_HOLD: u32 = 10;

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct PauseRow {
    /// Grouping key: source file (corpus) or utterance id (ring).
    pub file: String,
    pub idx: u32,
    /// Raw silence frames (to resume, or to the cut for the final pause).
    pub raw_len: u32,
    /// Max endpoint `consecutive_silence` (smoothed) reached in it.
    pub smoothed_len: u32,
    pub resumed: bool,
    /// Seconds into the utterance.
    pub at_s: f32,
    pub probes: Vec<(u32, [f32; NF])>,
    /// Ring only: the final pause was ended by the learned early cut.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub early: bool,
    /// Ring only: global row sequence number (monotonic).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub seq: u64,
}

fn is_zero(v: &u64) -> bool {
    *v == 0
}

impl PauseRow {
    /// A final pause that is a trustworthy END label. Corpus files end at
    /// chunk flushes as well as at real ends, so a corpus end counts only
    /// with ≥ 25 frames (750 ms) of trailing silence; ring ends are real
    /// endpoint cuts (relabelled when he resumed right after).
    pub fn is_clean_end(&self) -> bool {
        !self.resumed && (self.seq > 0 || self.raw_len >= 25)
    }

    /// Raw frame at which the unchanged endpoint ends this final pause.
    pub fn base_cut_frame(&self) -> u32 {
        if !self.early && self.smoothed_len >= BASE_FRAMES && self.raw_len >= self.smoothed_len {
            self.raw_len - (self.smoothed_len - BASE_FRAMES)
        } else {
            SMOOTHING_HOLD + BASE_FRAMES - 1
        }
    }
}

/// Builds the pause rows of ONE utterance on the capture thread. Feed it
/// every in-speech frame after `EotTracker::push`.
#[derive(Debug, Default)]
pub struct PauseLog {
    pub rows: Vec<PauseRow>,
    cur: Option<PauseRow>,
}

/// Rows kept per utterance (a 30-minute monologue cannot grow it).
const PAUSE_LOG_MAX: usize = 512;

impl PauseLog {
    pub fn reset(&mut self) {
        self.rows.clear();
        self.cur = None;
    }

    pub fn on_frame(&mut self, tracker: &EotTracker, raw_voiced: bool, smoothed_sil: u32, at_s: f32) {
        let s = tracker.raw_silence_frames();
        if raw_voiced {
            if let Some(mut p) = self.cur.take() {
                p.resumed = true;
                if !p.probes.is_empty() && self.rows.len() < PAUSE_LOG_MAX {
                    self.rows.push(p);
                }
            }
        } else if tracker.seen_voice() {
            let idx = self.rows.len() as u32;
            let p = self.cur.get_or_insert_with(|| PauseRow {
                idx,
                at_s,
                probes: Vec::with_capacity((PROBE_MAX - PROBE_MIN + 1) as usize),
                ..PauseRow::default()
            });
            p.raw_len = s;
            p.smoothed_len = p.smoothed_len.max(smoothed_sil);
            if (PROBE_MIN..=PROBE_MAX).contains(&s) && !tracker.held() {
                p.probes.push((s, tracker.features()));
            }
        }
    }

    /// The utterance ended at the endpoint: the open pause is its END.
    pub fn finish_end(&mut self, early: bool) {
        if let Some(mut p) = self.cur.take() {
            p.resumed = false;
            p.early = early;
            if !p.probes.is_empty() && self.rows.len() < PAUSE_LOG_MAX {
                self.rows.push(p);
            }
        }
    }

    /// Ended any other way (cap, recorder EOF, gate): no END label.
    pub fn finish_unlabelled(&mut self) {
        self.cur = None;
    }
}

// ---------------------------------------------------------------------
// Training: single-threaded logistic regression on standardized
// features, mini-batches, a caller-supplied pacing hook between batches.

/// Returned `false` aborts training (thermal rise, he came back, …).
pub type Pace<'a> = &'a mut dyn FnMut() -> bool;

#[derive(Clone, Debug)]
pub struct TrainCfg {
    /// Probe frames the model may cut at (raw silence).
    pub probes: std::ops::RangeInclusive<u32>,
    /// Threshold margins tried, smallest first; the first with zero
    /// held-out splits wins.
    pub margins: Vec<f32>,
    pub folds: u32,
    pub epochs: u32,
    pub batch: usize,
    /// A probe needs at least this many END and continuation rows.
    pub min_ends: usize,
    pub min_conts: usize,
}

impl Default for TrainCfg {
    fn default() -> Self {
        Self {
            probes: 6..=19,
            margins: vec![0.0, 0.25, 0.5, 0.75, 0.9],
            folds: 5,
            epochs: 40,
            batch: 32,
            min_ends: 8,
            min_conts: 30,
        }
    }
}

/// Fit one probe on (features, end?) pairs. Threshold = above the
/// highest continuation score in the training rows, `margin` of the way
/// to 1. `None` when there is not enough data or `pace` aborted.
pub fn fit_probe(
    xs: &[(&[f32; NF], bool)],
    probe_frames: u32,
    margin: f32,
    cfg: &TrainCfg,
    pace: Pace<'_>,
) -> Option<EotProbe> {
    let ends = xs.iter().filter(|(_, e)| *e).count();
    let conts = xs.len() - ends;
    if ends < cfg.min_ends || conts < cfg.min_conts {
        return None;
    }
    let n = xs.len() as f32;
    let mut mean = [0f32; NF];
    let mut std = [0f32; NF];
    for (f, _) in xs {
        for i in 0..NF {
            mean[i] += f[i] / n;
        }
    }
    for (f, _) in xs {
        for i in 0..NF {
            std[i] += (f[i] - mean[i]).powi(2) / n;
        }
    }
    for s in &mut std {
        *s = s.sqrt().max(1e-3);
    }
    // Class-balanced weights; L2; mini-batch gradient descent in a fixed
    // (deterministic) order.
    let pos_w = n / (2.0 * ends as f32);
    let neg_w = n / (2.0 * conts as f32);
    let mut w = [0f32; NF];
    let mut b = 0f32;
    let lr = 0.1f32;
    let mut order: Vec<usize> = (0..xs.len()).collect();
    let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ probe_frames as u64;
    for _ in 0..cfg.epochs {
        for i in (1..order.len()).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            order.swap(i, (rng % (i as u64 + 1)) as usize);
        }
        for batch in order.chunks(cfg.batch.max(1)) {
            let mut gw = [0f32; NF];
            let mut gb = 0f32;
            for &j in batch {
                let (f, end) = xs[j];
                let mut z = b;
                for i in 0..NF {
                    z += w[i] * (f[i] - mean[i]) / std[i];
                }
                let p = 1.0 / (1.0 + (-z).exp());
                let (y, cw) = if end { (1.0, pos_w) } else { (0.0, neg_w) };
                let g = cw * (p - y);
                for i in 0..NF {
                    gw[i] += g * (f[i] - mean[i]) / std[i];
                }
                gb += g;
            }
            let m = batch.len() as f32;
            for i in 0..NF {
                w[i] -= lr * (gw[i] / m + 0.05 * w[i]);
            }
            b -= lr * gb / m;
            if !pace() {
                return None;
            }
        }
    }
    let mut probe = EotProbe {
        probe_frames,
        mean: mean.to_vec(),
        std: std.to_vec(),
        w: w.to_vec(),
        b,
        threshold: 1.0,
    };
    let max_cont = xs
        .iter()
        .filter(|(_, e)| !*e)
        .map(|(f, _)| probe.score(f))
        .fold(0f32, f32::max);
    probe.threshold = (max_cont + (1.0 - max_cont) * margin).min(0.999_999);
    Some(probe)
}

/// Rows usable for training / evaluation: continuations, and clean ends.
pub fn labelled(rows: &[PauseRow]) -> impl Iterator<Item = &PauseRow> {
    rows.iter().filter(|r| r.resumed || r.is_clean_end())
}

/// Train one model (one probe per frame in `cfg.probes` with enough data).
pub fn train_model(
    rows: &[&PauseRow],
    margin: f32,
    cfg: &TrainCfg,
    pace: Pace<'_>,
) -> Option<EotModel> {
    let mut probes = Vec::new();
    let mut xs: Vec<(&[f32; NF], bool)> = Vec::with_capacity(rows.len());
    for k in cfg.probes.clone() {
        xs.clear();
        for r in rows {
            if let Some((_, f)) = r.probes.iter().find(|(p, _)| *p == k) {
                xs.push((f, !r.resumed));
            }
        }
        match fit_probe(&xs, k, margin, cfg, pace) {
            Some(p) => probes.push(p),
            None if !pace() => return None,
            None => {}
        }
    }
    let m = EotModel {
        probes,
        trained_through_seq: rows.iter().map(|r| r.seq).max().unwrap_or(0),
        ..EotModel::default()
    };
    m.is_valid().then_some(m)
}

/// Replay of a model (or the unchanged endpoint, `None`) over pause rows.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct EvalReport {
    /// Resumed (mid-thought) pauses.
    pub continuations: usize,
    /// Of those, cut by the learned model (new splits it causes).
    pub model_splits: usize,
    /// Of those, cut by the unchanged base window (≥ 0.6 s smoothed).
    pub base_splits: usize,
    pub ends: usize,
    /// Ends the model cut before the base window.
    pub early_ends: usize,
    /// Raw frames from his last voiced frame to the cut, per end.
    pub end_cut_frames: Vec<u32>,
}

impl EvalReport {
    pub fn add(&mut self, o: &EvalReport) {
        self.continuations += o.continuations;
        self.model_splits += o.model_splits;
        self.base_splits += o.base_splits;
        self.ends += o.ends;
        self.early_ends += o.early_ends;
        self.end_cut_frames.extend_from_slice(&o.end_cut_frames);
    }

    pub fn pct_ms(&self, q: f64) -> Option<u32> {
        let mut v = self.end_cut_frames.clone();
        if v.is_empty() {
            return None;
        }
        v.sort_unstable();
        Some(v[((v.len() as f64 - 1.0) * q).round() as usize] * FRAME_MS)
    }
}

pub fn evaluate<'a>(model: Option<&EotModel>, rows: impl Iterator<Item = &'a PauseRow>) -> EvalReport {
    let mut r = EvalReport::default();
    for row in rows {
        let cut = model.and_then(|m| m.first_cut(row));
        if row.resumed {
            r.continuations += 1;
            r.base_splits += usize::from(row.smoothed_len >= BASE_FRAMES);
            r.model_splits += usize::from(cut.is_some_and(|k| k <= row.raw_len));
        } else if row.is_clean_end() {
            r.ends += 1;
            let base = row.base_cut_frame();
            match cut {
                Some(k) if k < base => {
                    r.early_ends += 1;
                    r.end_cut_frames.push(k);
                }
                _ => r.end_cut_frames.push(base),
            }
        }
    }
    r
}

/// Deterministic fold of a group key (FNV-1a).
pub fn fold_of(key: &str, folds: u32) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in key.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h % folds.max(1)
}

/// Result of grouped cross-validation + the final model.
#[derive(Clone, Debug)]
pub struct Selected {
    pub margin: f32,
    /// Held-out replay of the chosen margin (every row predicted by a
    /// model that never saw its utterance / file).
    pub cv: EvalReport,
    pub model: EotModel,
}

/// Grouped K-fold CV over margins (each fold's probes are fitted once;
/// margins only move thresholds), pick the smallest margin with ZERO
/// held-out splits and at least one early end, then fit on all rows.
pub fn select_and_train(rows: &[PauseRow], cfg: &TrainCfg, pace: Pace<'_>) -> Option<Selected> {
    let lab: Vec<&PauseRow> = labelled(rows).collect();
    let mut per_margin: Vec<EvalReport> = vec![EvalReport::default(); cfg.margins.len()];
    for fold in 0..cfg.folds {
        let (train, test): (Vec<&PauseRow>, Vec<&PauseRow>) =
            lab.iter().partition(|r| fold_of(&r.file, cfg.folds) != fold);
        let Some(base) = train_model(&train, 0.0, cfg, pace) else {
            if !pace() {
                return None;
            }
            for rep in per_margin.iter_mut() {
                rep.add(&evaluate(None, test.iter().copied()));
            }
            continue;
        };
        // Recompute the zero-margin max continuation score per probe, then
        // re-threshold for every margin without refitting.
        for (mi, &margin) in cfg.margins.iter().enumerate() {
            let mut m = base.clone();
            for p in &mut m.probes {
                let mc = train_max_cont(&train, p);
                p.threshold = (mc + (1.0 - mc) * margin).min(0.999_999);
            }
            per_margin[mi].add(&evaluate(Some(&m), test.iter().copied()));
        }
    }
    let (mi, cv) = per_margin
        .into_iter()
        .enumerate()
        .find(|(_, rep)| rep.model_splits == 0 && rep.early_ends > 0)?;
    let margin = cfg.margins[mi];
    let model = train_model(&lab, margin, cfg, pace)?;
    Some(Selected { margin, cv, model })
}

fn train_max_cont(train: &[&PauseRow], p: &EotProbe) -> f32 {
    train
        .iter()
        .filter(|r| r.resumed)
        .filter_map(|r| r.probes.iter().find(|(k, _)| *k == p.probe_frames))
        .map(|(_, f)| p.score(f))
        .fold(0f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f32, amp: f32) -> Vec<i16> {
        (0..480)
            .map(|i| (amp * (2.0 * std::f32::consts::PI * hz * i as f32 / 16000.0).sin()) as i16)
            .collect()
    }

    #[test]
    fn pitch_tracks_a_tone() {
        let st = pitch_st(&tone(110.0, 8000.0)).unwrap();
        assert!((st - 12.0).abs() < 0.6, "{st}"); // 110 Hz = 12 st re 55
        assert!(pitch_st(&[0i16; 480]).is_none());
    }

    #[test]
    fn falling_pitch_and_energy_show_in_features() {
        let mut t = EotTracker::default();
        for i in 0..40 {
            t.push(&tone(140.0, 9000.0), 0.9, i % 7 != 6);
        }
        for i in 0..8 {
            t.push(&tone(140.0 - 6.0 * i as f32, 9000.0 - 900.0 * i as f32), 0.9, true);
        }
        for _ in 0..5 {
            t.push(&[0i16; 480], 0.02, false);
        }
        assert_eq!(t.raw_silence_frames(), 5);
        let f = t.features();
        assert!(f[1] < 0.0, "energy slope {f:?}");
        assert!(f[2] < 0.0, "pitch slope {f:?}");
        assert!(f[3] < 0.0, "final pitch below median {f:?}");
    }

    #[test]
    fn model_without_matching_probe_never_cuts() {
        let m = EotModel::default();
        assert!(!m.is_valid());
        let t = EotTracker::default();
        assert!(m.cut_now(&t).is_none());
    }

    fn features_reference(t: &EotTracker) -> [f32; NF] {
        let s = t;
        let sil = s.raw_silence as usize;
        let n = s.ring.len();
        let pause: Vec<&Obs> = s.ring.iter().skip(n.saturating_sub(sil)).collect();
        let voiced: Vec<&Obs> = s
            .ring
            .iter()
            .take(n.saturating_sub(sil))
            .filter(|o| o.voiced)
            .collect();
        let med_db = hist_median(&s.db_hist).map_or(-30.0, |b| b * 1.25 - 80.0);
        let med_st = hist_median(&s.st_hist);
        let tail = &voiced[voiced.len().saturating_sub(8)..];
        let last3 = &voiced[voiced.len().saturating_sub(3)..];
        let final_db = if last3.is_empty() {
            med_db
        } else {
            last3.iter().map(|o| o.db).sum::<f32>() / last3.len() as f32
        };
        let dbs: Vec<f32> = tail.iter().map(|o| o.db).collect();
        let sts: Vec<f32> = tail.iter().filter(|o| o.st > 0.0).map(|o| o.st).collect();
        let st3: Vec<f32> = last3.iter().filter(|o| o.st > 0.0).map(|o| o.st).collect();
        let final_st_rel = match (med_st, st3.is_empty()) {
            (Some(m), false) => st3.iter().sum::<f32>() / st3.len() as f32 - (m + 0.5),
            _ => 0.0,
        };
        let mean_run = if s.run_count > 0 {
            s.run_len_sum as f32 / s.run_count as f32
        } else {
            1.0
        };
        let secs = (s.frames as f32 * 0.03).max(0.3);
        let breath = pause
            .iter()
            .map(|o| o.db)
            .fold(f32::MIN, f32::max)
            .max(-100.0);
        let sil_prob = if pause.is_empty() {
            0.0
        } else {
            pause.iter().map(|o| o.prob).sum::<f32>() / pause.len() as f32
        };
        let voicing = if tail.is_empty() {
            0.0
        } else {
            sts.len() as f32 / tail.len() as f32
        };
        [
            final_db - med_db,
            slope(&dbs),
            if sts.len() >= 3 { slope(&sts) } else { 0.0 },
            final_st_rel,
            s.last_run_len as f32 / mean_run.max(1.0),
            secs.ln(),
            s.runs as f32 / secs,
            s.pauses as f32 / secs,
            breath - med_db,
            sil_prob,
            voicing,
        ]
    }

    /// The allocation-free `features` must equal the original
    /// implementation the corpus rows were extracted with.
    #[test]
    fn features_match_reference_implementation() {
        let mut t = EotTracker::default();
        let mut x = 12345u32;
        for i in 0..400 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let voiced = (x % 7) > 1 || i % 37 < 3;
            let hz = 90.0 + (x % 120) as f32;
            let amp = if voiced { 2000.0 + (x % 9000) as f32 } else { 30.0 };
            let prob = if voiced { 0.6 + (x % 40) as f32 / 100.0 } else { (x % 30) as f32 / 100.0 };
            t.push(&tone(hz, amp), prob, voiced);
            if t.raw_silence_frames() > 0 {
                let (a, b) = (t.features(), features_reference(&t));
                for k in 0..NF {
                    assert!((a[k] - b[k]).abs() <= 1e-4 * (1.0 + b[k].abs()), "frame {i} f{k}: {a:?} vs {b:?}");
                }
            }
        }
    }

    fn row(file: &str, v: f32, end: bool, raw_len: u32) -> PauseRow {
        PauseRow {
            file: file.into(),
            raw_len,
            smoothed_len: if end { 25 } else { 0 },
            resumed: !end,
            probes: (PROBE_MIN..=raw_len.min(PROBE_MAX))
                .map(|k| {
                    let mut f = [0f32; NF];
                    f[0] = v;
                    f[9] = k as f32 * 0.01;
                    (k, f)
                })
                .collect(),
            ..PauseRow::default()
        }
    }

    #[test]
    fn fit_probe_threshold_cuts_no_training_continuation() {
        let rows: Vec<PauseRow> = (0..120)
            .map(|i| row(&format!("f{i}"), i as f32 / 30.0, i >= 90, if i >= 90 { 30 } else { 20 }))
            .collect();
        let refs: Vec<&PauseRow> = rows.iter().collect();
        let cfg = TrainCfg::default();
        let m = train_model(&refs, 0.0, &cfg, &mut || true).unwrap();
        let rep = evaluate(Some(&m), rows.iter());
        assert_eq!(rep.model_splits, 0, "{rep:?}");
        assert!(rep.early_ends > 0, "{rep:?}");
        assert!(m.is_valid());
    }

    #[test]
    fn select_needs_zero_heldout_splits_and_pace_can_abort() {
        let rows: Vec<PauseRow> = (0..200)
            .map(|i| row(&format!("f{i}"), i as f32 / 50.0, i >= 150, if i >= 150 { 30 } else { 22 }))
            .collect();
        let cfg = TrainCfg::default();
        let sel = select_and_train(&rows, &cfg, &mut || true).expect("separable data selects");
        assert_eq!(sel.cv.model_splits, 0);
        assert!(sel.cv.early_ends > 0);
        let mut calls = 0;
        assert!(select_and_train(&rows, &cfg, &mut || {
            calls += 1;
            calls < 5
        })
        .is_none());
        // Inseparable labels: nothing is selected, no model, no early cut.
        let noisy: Vec<PauseRow> = (0..200)
            .map(|i| row(&format!("g{i}"), (i % 10) as f32, i % 4 == 0, if i % 4 == 0 { 30 } else { 22 }))
            .collect();
        assert!(select_and_train(&noisy, &cfg, &mut || true).is_none());
    }

    #[test]
    fn pause_log_labels_resumed_and_final_pauses() {
        let mut t = EotTracker::default();
        let mut log = PauseLog::default();
        let feed = |t: &mut EotTracker, log: &mut PauseLog, voiced: bool| {
            let (s, p) = if voiced { (tone(130.0, 8000.0), 0.9) } else { (vec![0i16; 480], 0.01) };
            t.push(&s, p, voiced);
            log.on_frame(t, voiced, 0, 0.0);
        };
        for _ in 0..20 { feed(&mut t, &mut log, true); }
        for _ in 0..2 { feed(&mut t, &mut log, false); } // too short: no row
        for _ in 0..10 { feed(&mut t, &mut log, true); }
        for _ in 0..12 { feed(&mut t, &mut log, false); }
        for _ in 0..10 { feed(&mut t, &mut log, true); }
        for _ in 0..9 { feed(&mut t, &mut log, false); }
        log.finish_end(true);
        assert_eq!(log.rows.len(), 2, "{:?}", log.rows.iter().map(|r| r.raw_len).collect::<Vec<_>>());
        assert!(log.rows[0].resumed && log.rows[0].raw_len == 12);
        assert_eq!(log.rows[0].probes.len(), 10); // frames 3..=12
        assert!(!log.rows[1].resumed && log.rows[1].early && log.rows[1].raw_len == 9);
        // A held pause records no probe and cannot be cut.
        t.hold();
        assert!(t.held());
    }
}

