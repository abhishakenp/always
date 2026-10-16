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

use super::eot::{EotTracker, PauseLog, PauseRow, VOICED_PROB};
use super::vad_silero::SileroVad;
use std::collections::VecDeque;

const SPEECH_THRESHOLD: f32 = 0.5;
const SILENCE_THRESHOLD: f32 = SPEECH_THRESHOLD * 0.75;
const ENERGY_THRESHOLD: f64 = 0.012;
const MIN_SPEECH_FRAMES: usize = 2; // onset 60 ms
const SMOOTHING: usize = 10;

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

/// Runs one file; emits one row per pause (≥ `PROBE_MIN` raw frames).
pub(crate) fn pauses_of(vad: &SileroVad, file: &str, samples: &[i16]) -> Vec<PauseRow> {
    let mut hist: VecDeque<f32> = VecDeque::new();
    let mut tracker = EotTracker::default();
    let mut log = PauseLog::default();
    let (mut in_speech, mut cons_speech, mut cons_sil) = (false, 0usize, 0usize);
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
        log.on_frame(&tracker, raw_voiced, cons_sil as u32, fi as f32 * 0.03);
    }
    log.finish_end(false);
    for r in &mut log.rows {
        r.file = file.to_string();
    }
    log.rows
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
    use super::eot::{EotModel, EotProbe, NF};
    let frame: Vec<i16> = (0..480)
        .map(|i| (8000.0 * (2.0 * std::f32::consts::PI * 130.0 * i as f32 / 16000.0).sin()) as i16)
        .collect();
    let mut t = EotTracker::default();
    let start = std::time::Instant::now();
    for i in 0..1000 {
        t.push(&frame, 0.9, i % 9 != 1);
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
        ..EotModel::default()
    };
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        std::hint::black_box(m.cut_now(&t));
    }
    let probe_us = start.elapsed().as_micros() as f64 / 1000.0;
    assert_eq!(t.raw_silence_frames(), 8);
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        std::hint::black_box(t.features());
    }
    let feat_us = start.elapsed().as_micros() as f64 / 1000.0;
    let mut log = super::eot::PauseLog::default();
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        log.on_frame(&t, false, 0, 1.0);
    }
    let row_us = start.elapsed().as_micros() as f64 / 1000.0;
    eprintln!(
        "eot cost: push {push_us:.1} µs/frame, cut_now {probe_us:.2} µs (fired {:?}), features {feat_us:.2} µs, pause-row frame {row_us:.2} µs",
        m.cut_now(&t)
    );
    assert!(push_us < 1000.0 && probe_us < 1000.0);
}

fn load_rows(path: &str) -> Vec<PauseRow> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter_map(|l| serde_json::from_str::<PauseRow>(l).ok())
        .collect()
}

fn peak_rss_mb() -> f64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    ru.ru_maxrss as f64 / (1024.0 * 1024.0) // bytes on macOS
}

fn cpu_secs() -> f64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    ru.ru_utime.tv_sec as f64 + ru.ru_utime.tv_usec as f64 / 1e6 + ru.ru_stime.tv_sec as f64
        + ru.ru_stime.tv_usec as f64 / 1e6
}

/// Offline replay of the learned end-of-turn over his audio pause rows
/// (grouped CV: every row scored by a model that never saw its file).
///
/// `EOT_ROWS=<rows.jsonl> [EOT_MODEL_OUT=<model.json>] cargo test --lib
/// always::eot_corpus::eot_replay -- --ignored --nocapture`
#[test]
#[ignore = "needs EOT_ROWS=<rows.jsonl>; run explicitly"]
fn eot_replay() {
    use super::eot::{TrainCfg, evaluate, labelled, select_and_train};
    let Ok(path) = std::env::var("EOT_ROWS") else {
        return;
    };
    let rows = load_rows(&path);
    let rss0 = peak_rss_mb();
    let base = evaluate(None, labelled(&rows));
    eprintln!(
        "rows {}  continuations {}  clean ends {}  base splits {}  base end cut p50 {:?} p90 {:?} ms",
        rows.len(),
        base.continuations,
        base.ends,
        base.base_splits,
        base.pct_ms(0.5),
        base.pct_ms(0.9)
    );
    let mut best = None;
    for probes in [3..=19u32, 6..=19, 7..=10, 7..=19, 8..=19, 10..=19, 13..=19] {
        let cfg = TrainCfg {
            probes: probes.clone(),
            ..TrainCfg::default()
        };
        let (t0, c0) = (std::time::Instant::now(), cpu_secs());
        let mut batches = 0u64;
        let sel = select_and_train(&rows, &cfg, &mut || {
            batches += 1;
            true
        });
        let (wall, cpu) = (t0.elapsed().as_secs_f64(), cpu_secs() - c0);
        match &sel {
            Some(s) => eprintln!(
                "probes {probes:?}: margin {:.2}  CV model splits {}/{}  early ends {}/{}  end cut p50 {:?} p90 {:?} ms  | train wall {:.2}s cpu {:.2}s batches {batches}",
                s.margin,
                s.cv.model_splits,
                s.cv.continuations,
                s.cv.early_ends,
                s.cv.ends,
                s.cv.pct_ms(0.5),
                s.cv.pct_ms(0.9),
                wall,
                cpu
            ),
            None => eprintln!("probes {probes:?}: no margin gives zero held-out splits with an early end (train wall {wall:.2}s cpu {cpu:.2}s)"),
        }
        if let Some(s) = &sel {
            let early: Vec<u32> = s.cv.end_cut_frames.iter().filter(|&&f| f < 29).map(|f| f * 30).collect();
            let ins = evaluate(Some(&s.model), labelled(&rows));
            eprintln!(
                "    held-out early cuts at {early:?} ms; final model in-sample: splits {}/{} early ends {}/{}",
                ins.model_splits, ins.continuations, ins.early_ends, ins.ends
            );
        }
        if let Some(s) = sel
            && best.as_ref().is_none_or(|b: &super::eot::Selected| s.cv.early_ends > b.cv.early_ends)
        {
            best = Some(s);
        }
    }
    eprintln!("peak RSS {:.1} MB (before training {rss0:.1} MB)", peak_rss_mb());
    if let (Some(b), Ok(out)) = (best, std::env::var("EOT_MODEL_OUT")) {
        std::fs::write(&out, serde_json::to_string_pretty(&b.model).unwrap()).unwrap();
        eprintln!("model → {out} ({} probes)", b.model.probes.len());
    }
}

/// The daemon's retrain-when-away pass, run once over a learn dir seeded
/// with his corpus rows: decision, wall / CPU / sleep time, thermal, RSS.
///
/// `EOT_LEARN_DIR=<dir with seed-rows.jsonl> cargo test --lib
/// always::eot_corpus::eot_retrain_once -- --ignored --nocapture`
#[test]
#[ignore = "needs EOT_LEARN_DIR=<dir>; run explicitly"]
fn eot_retrain_once() {
    use super::learn_train::{Pacer, retrain_once};
    let Ok(d) = std::env::var("EOT_LEARN_DIR") else {
        return;
    };
    super::learn::background_qos();
    let rss0 = peak_rss_mb();
    let mut pacer = Pacer::new(0.2, Box::new(|| None));
    let o = retrain_once(std::path::Path::new(&d), &super::eot::TrainCfg::default(), &mut pacer).unwrap();
    eprintln!(
        "rows {}  decision {:?}  cv {:?}  current held-out splits {:?}",
        o.rows,
        o.decision,
        o.candidate.as_ref().map(|(m, r)| (m, r.model_splits, r.continuations, r.early_ends, r.ends, r.pct_ms(0.5), r.pct_ms(0.9))),
        o.current_heldout.as_ref().map(|r| (r.model_splits, r.continuations)),
    );
    eprintln!(
        "wall {} ms  thread cpu {} ms  slept {} ms  duty {:.2}  thermal max {:?}  peak RSS {:.1} MB (before {rss0:.1} MB)",
        o.wall.as_millis(),
        o.cpu.as_millis(),
        o.slept.as_millis(),
        o.cpu.as_secs_f64() / o.wall.as_secs_f64().max(1e-9),
        o.thermal_max,
        peak_rss_mb()
    );
}
