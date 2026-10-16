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
    /// `raw_silence_frames() > 0`).
    pub fn features(&self) -> [f32; NF] {
        let sil = self.raw_silence as usize;
        let n = self.ring.len();
        let pause: Vec<&Obs> = self.ring.iter().skip(n.saturating_sub(sil)).collect();
        let voiced: Vec<&Obs> = self
            .ring
            .iter()
            .take(n.saturating_sub(sil))
            .filter(|o| o.voiced)
            .collect();
        let med_db = hist_median(&self.db_hist).map_or(-30.0, |b| b * 1.25 - 80.0);
        let med_st = hist_median(&self.st_hist);
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
        let mean_run = if self.run_count > 0 {
            self.run_len_sum as f32 / self.run_count as f32
        } else {
            1.0
        };
        let secs = (self.frames as f32 * 0.03).max(0.3);
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
}

impl EotModel {
    /// Valid shape (feature count, ordering) — a malformed file is ignored.
    pub fn is_valid(&self) -> bool {
        !self.probes.is_empty()
            && self.probes.iter().all(|p| {
                p.w.len() == NF && p.mean.len() == NF && p.std.len() == NF && p.threshold.is_finite()
            })
    }

    /// At exactly a probe frame: does the model say he is done?
    pub fn cut_now(&self, tracker: &EotTracker) -> Option<f32> {
        let s = tracker.raw_silence_frames();
        let p = self.probes.iter().find(|p| p.probe_frames == s)?;
        let score = p.score(&tracker.features());
        (score > p.threshold).then_some(score)
    }
}

/// `~/Library/Application Support/always/eot-model.json`.
pub fn model_path() -> Option<std::path::PathBuf> {
    dirs::data_dir().map(|d| d.join("always").join("eot-model.json"))
}

/// Load the per-user model; `None` (no early cuts) when absent/invalid.
pub fn load_model() -> Option<EotModel> {
    let text = std::fs::read_to_string(model_path()?).ok()?;
    let m: EotModel = serde_json::from_str(&text).ok()?;
    m.is_valid().then_some(m)
}

// ---------------------------------------------------------------------
// Training: single-threaded logistic regression on standardized features.

/// One labelled probe row: features at `probe_frames` of raw silence of
/// a pause that either ended the utterance (`end`) or resumed.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EotRow {
    pub probe_frames: u32,
    pub f: Vec<f32>,
    pub end: bool,
    /// Grouping key for held-out validation (file / day).
    #[serde(default)]
    pub group: String,
}

/// Fit one probe. Threshold = just above the highest continuation score
/// in the training rows (zero training splits), times `margin` toward 1.
pub fn fit_probe(rows: &[&EotRow], probe_frames: u32, margin: f32) -> Option<EotProbe> {
    let ends = rows.iter().filter(|r| r.end).count();
    if ends < 5 || ends == rows.len() {
        return None;
    }
    let n = rows.len() as f32;
    let mut mean = vec![0f32; NF];
    let mut std = vec![0f32; NF];
    for r in rows {
        for i in 0..NF {
            mean[i] += r.f[i] / n;
        }
    }
    for r in rows {
        for i in 0..NF {
            std[i] += (r.f[i] - mean[i]).powi(2) / n;
        }
    }
    for s in &mut std {
        *s = s.sqrt().max(1e-3);
    }
    let xs: Vec<Vec<f32>> = rows
        .iter()
        .map(|r| (0..NF).map(|i| (r.f[i] - mean[i]) / std[i]).collect())
        .collect();
    // Class-balanced weights; L2 0.05; plain gradient descent.
    let pos_w = n / (2.0 * ends as f32);
    let neg_w = n / (2.0 * (rows.len() - ends) as f32);
    let mut w = vec![0f32; NF];
    let mut b = 0f32;
    let lr = 0.1;
    for _ in 0..600 {
        let mut gw = vec![0f32; NF];
        let mut gb = 0f32;
        for (x, r) in xs.iter().zip(rows) {
            let z = b + x.iter().zip(&w).map(|(a, c)| a * c).sum::<f32>();
            let p = 1.0 / (1.0 + (-z).exp());
            let (y, cw) = if r.end { (1.0, pos_w) } else { (0.0, neg_w) };
            let g = cw * (p - y);
            for i in 0..NF {
                gw[i] += g * x[i];
            }
            gb += g;
        }
        for i in 0..NF {
            w[i] -= lr * (gw[i] / n + 0.05 * w[i]);
        }
        b -= lr * gb / n;
    }
    let mut probe = EotProbe {
        probe_frames,
        mean,
        std,
        w,
        b,
        threshold: 1.0,
    };
    let max_cont = rows
        .iter()
        .filter(|r| !r.end)
        .map(|r| probe.score(r.f.as_slice().try_into().unwrap()))
        .fold(0f32, f32::max);
    probe.threshold = (max_cont + (1.0 - max_cont) * margin).min(0.999_999);
    Some(probe)
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

    #[test]
    fn fit_probe_threshold_cuts_no_training_continuation() {
        let mk = |v: f32, end: bool| EotRow {
            probe_frames: 8,
            f: (0..NF).map(|i| if i == 0 { v } else { 0.1 * i as f32 }).collect(),
            end,
            group: String::new(),
        };
        let rows: Vec<EotRow> = (0..40)
            .map(|i| mk(i as f32 / 10.0, i >= 30))
            .chain((0..5).map(|i| mk(2.5 + i as f32 * 0.05, false)))
            .collect();
        let refs: Vec<&EotRow> = rows.iter().collect();
        let p = fit_probe(&refs, 8, 0.0).unwrap();
        for r in &rows {
            let s = p.score(r.f.as_slice().try_into().unwrap());
            if !r.end {
                assert!(s <= p.threshold, "continuation cut");
            }
        }
        let cut_ends = rows
            .iter()
            .filter(|r| r.end && p.score(r.f.as_slice().try_into().unwrap()) > p.threshold)
            .count();
        assert!(cut_ends > 0);
    }
}
