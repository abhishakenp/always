//! Offline extraction of end-of-turn rows from his own recordings, through
//! the SAME Silero + energy + `EotTracker` the daemon runs per frame.
//!
//! `EOT_CORPUS=<dir>[:<dir>…] EOT_OUT=<rows.jsonl> cargo test --lib
//! always::eot_corpus -- --ignored --nocapture`
//!
//! Every raw silence run after voice is one pause: `resumed` when speech
//! follows inside the file, otherwise the file's final pause (its length
//! is the trailing silence). Per pause the extractor writes the raw
//! length, the endpoint's smoothed silence count (what the base 0.6 s
//! window measures) and the tracker's features at each probe frame the
//! pause reached.

use super::eot::{EotTracker, NF, VOICED_PROB};
use super::vad_silero::SileroVad;
use std::collections::VecDeque;

const SPEECH_THRESHOLD: f32 = 0.5;
const SILENCE_THRESHOLD: f32 = SPEECH_THRESHOLD * 0.75;
const ENERGY_THRESHOLD: f64 = 0.012;
const MIN_SPEECH_FRAMES: usize = 2; // onset 60 ms
const SMOOTHING: usize = 10;
pub const PROBES: std::ops::RangeInclusive<u32> = 3..=24;

fn read_wav_16k_mono(path: &std::path::Path) -> Option<Vec<i16>> {
    let b = std::fs::read(path).ok()?;
    if b.len() < 44 || &b[0..4] != b"RIFF" {
        return None;
    }
    let mut i = 12;
    let (mut ch, mut sr, mut bits) = (0u16, 0u32, 0u16);
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let len = u32::from_le_bytes(b[i + 4..i + 8].try_into().ok()?) as usize;
        let body = i + 8;
        if id == b"fmt " {
            ch = u16::from_le_bytes(b[body + 2..body + 4].try_into().ok()?);
            sr = u32::from_le_bytes(b[body + 4..body + 8].try_into().ok()?);
            bits = u16::from_le_bytes(b[body + 14..body + 16].try_into().ok()?);
        } else if id == b"data" {
            if ch != 1 || sr != 16_000 || bits != 16 {
                return None;
            }
            let end = (body + len).min(b.len());
            return Some(
                b[body..end]
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect(),
            );
        }
        i = body + len + (len & 1);
    }
    None
}

fn energy(samples: &[i16]) -> f64 {
    let ss: i64 = samples.iter().map(|&s| s as i64 * s as i64).sum();
    (ss as f64 / samples.len() as f64).sqrt() / 32768.0
}

#[derive(serde::Serialize)]
struct PauseRow {
    file: String,
    idx: u32,
    /// Raw silence frames (to resume, or trailing for the final pause).
    raw_len: u32,
    /// Max endpoint `consecutive_silence` (smoothed) reached in it.
    smoothed_len: u32,
    resumed: bool,
    /// Seconds of the file before this pause.
    at_s: f32,
    probes: Vec<(u32, [f32; NF])>,
}

/// Runs one file; emits one row per pause.
fn pauses_of(vad: &SileroVad, file: &str, samples: &[i16]) -> Vec<PauseRow> {
    let mut out = Vec::new();
    let mut hist: VecDeque<f32> = VecDeque::new();
    let mut tracker = EotTracker::default();
    let (mut in_speech, mut cons_speech, mut cons_sil) = (false, 0usize, 0usize);
    let mut cur: Option<PauseRow> = None;
    for (fi, frame) in samples.chunks_exact(480).enumerate() {
        let prob = vad.predict(frame).unwrap_or(0.0);
        if hist.len() >= SMOOTHING {
            hist.pop_front();
        }
        hist.push_back(prob);
        let smax = hist.iter().copied().fold(0f32, f32::max);
        let e = energy(frame);
        let loud = e >= ENERGY_THRESHOLD * 2.0;
        let is_speech = if in_speech {
            smax >= SILENCE_THRESHOLD || loud
        } else {
            prob >= SPEECH_THRESHOLD || (prob >= SPEECH_THRESHOLD * 0.65 && e >= ENERGY_THRESHOLD * 1.1)
        };
        if is_speech {
            cons_speech += 1;
            cons_sil = 0;
            if cons_speech >= MIN_SPEECH_FRAMES {
                in_speech = true;
            }
        } else if in_speech {
            cons_sil += 1;
            cons_speech = 0;
        }
        if !in_speech {
            continue;
        }
        let raw_voiced = prob >= VOICED_PROB || loud;
        tracker.push(frame, prob, raw_voiced);
        let s = tracker.raw_silence_frames();
        if raw_voiced {
            if let Some(mut p) = cur.take() {
                p.resumed = true;
                out.push(p);
            }
        } else if tracker.seen_voice() {
            let p = cur.get_or_insert_with(|| PauseRow {
                file: file.to_string(),
                idx: out.len() as u32,
                raw_len: 0,
                smoothed_len: 0,
                resumed: false,
                at_s: fi as f32 * 0.03,
                probes: Vec::new(),
            });
            p.raw_len = s;
            p.smoothed_len = p.smoothed_len.max(cons_sil as u32);
            if PROBES.contains(&s) {
                p.probes.push((s, tracker.features()));
            }
        }
    }
    if let Some(p) = cur.take() {
        out.push(p);
    }
    out
}

#[test]
#[ignore = "needs EOT_CORPUS=<dirs> EOT_OUT=<file>; run explicitly"]
fn eot_corpus_extract() {
    let (Ok(dirs), Ok(outp)) = (std::env::var("EOT_CORPUS"), std::env::var("EOT_OUT")) else {
        return;
    };
    let vad = SileroVad::new().unwrap();
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    for d in dirs.split(':') {
        for e in walk(std::path::Path::new(d)) {
            if e.extension().is_some_and(|x| x == "wav") {
                files.push(e);
            }
        }
    }
    files.sort();
    let mut w = String::new();
    let (mut n_files, mut n_pauses) = (0, 0);
    for f in &files {
        let Some(s) = read_wav_16k_mono(f) else {
            continue;
        };
        n_files += 1;
        for p in pauses_of(&vad, &f.to_string_lossy(), &s) {
            n_pauses += 1;
            w.push_str(&serde_json::to_string(&p).unwrap());
            w.push('\n');
        }
    }
    std::fs::write(&outp, w).unwrap();
    eprintln!("files {n_files} pauses {n_pauses} → {outp}");
}

fn walk(d: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut v = Vec::new();
    if let Ok(rd) = std::fs::read_dir(d) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                v.extend(walk(&p));
            } else {
                v.push(p);
            }
        }
    }
    v
}

/// Per-decision runtime cost: one voiced frame push + a probe score.
#[test]
fn eot_per_frame_and_probe_cost_is_tiny() {
    use super::eot::{EotModel, EotProbe};
    let frame: Vec<i16> = (0..480)
        .map(|i| (8000.0 * (2.0 * std::f32::consts::PI * 130.0 * i as f32 / 16000.0).sin()) as i16)
        .collect();
    let mut t = EotTracker::default();
    let start = std::time::Instant::now();
    for i in 0..1000 {
        t.push(&frame, 0.9, i % 9 != 0);
    }
    let push_us = start.elapsed().as_micros() as f64 / 1000.0;
    for _ in 0..8 {
        t.push(&[0; 480], 0.01, false);
    }
    let m = EotModel {
        probes: vec![EotProbe {
            probe_frames: 8,
            mean: vec![0.0; NF],
            std: vec![1.0; NF],
            w: vec![0.1; NF],
            b: 0.0,
            threshold: 0.5,
        }],
        note: String::new(),
    };
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        std::hint::black_box(m.cut_now(&t));
    }
    let probe_us = start.elapsed().as_micros() as f64 / 1000.0;
    eprintln!("eot cost: push {push_us:.1} µs/frame, probe {probe_us:.1} µs");
    assert!(push_us < 1000.0 && probe_us < 1000.0);
}
