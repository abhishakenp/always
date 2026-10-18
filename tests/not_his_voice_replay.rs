//! Replays every speaker-gate-verified utterance of 2026-10-15..18 (his real
//! daemon logs, reduced to numbers in `tests/fixtures/gate_replay.tsv`)
//! through the paste decision `hallucination::not_his_voice`, against the
//! hand labels (every transcript read: H his, F not his, U unsure).
//!
//! Measures — never assumes — the false-hold rate on his speech and the
//! catch rate on speech that is not his, and sweeps the bar over every
//! observed score to show where his profile threshold sits.
//!
//! `logged_outcome = foreign_speech_rejected` rows were already held by the
//! live-recognizer rule (rule 2 of `foreign_speech`), which this change keeps.

use always::always::hallucination::not_his_voice;
use always::stt::{RecognitionEvidence, TranscriptionResult};

/// His profile threshold in these logs (`speaker_gate_threshold`, logged as
/// `"threshold":0.5`).
const BAR: f32 = 0.50;

struct Row {
    best: f32,
    conf: Option<f32>,
    already_held: bool,
    label: char,
}

fn rows() -> Vec<Row> {
    let body = include_str!("fixtures/gate_replay.tsv");
    body.lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("ts\t"))
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            Row {
                best: f[2].parse().unwrap(),
                conf: f[3].parse().ok(),
                already_held: f[5] == "foreign_speech_rejected",
                label: f[6].chars().next().unwrap(),
            }
        })
        .collect()
}

fn held(r: &Row) -> bool {
    held_at(r, BAR)
}

fn held_at(r: &Row, bar: f32) -> bool {
    let t = TranscriptionResult {
        text: "words".into(),
        evidence: RecognitionEvidence {
            confidence: r.conf,
            voice_best: Some(r.best),
            voice_bar: Some(bar),
            ..Default::default()
        },
        ..Default::default()
    };
    r.already_held || not_his_voice(&t).is_some()
}

#[test]
fn replay_his_utterances_through_the_voice_hold() {
    let rows = rows();
    let his: Vec<&Row> = rows.iter().filter(|r| r.label == 'H').collect();
    let not_his: Vec<&Row> = rows.iter().filter(|r| r.label == 'F').collect();
    let unsure: Vec<&Row> = rows.iter().filter(|r| r.label == 'U').collect();
    let false_holds = his.iter().filter(|r| held(r)).count();
    let caught = not_his.iter().filter(|r| held(r)).count();
    let before = not_his.iter().filter(|r| r.already_held).count();
    let unsure_held = unsure.iter().filter(|r| held(r)).count();
    println!(
        "his speech held {false_holds}/{} ({:.2}%); not-his caught {caught}/{} ({:.1}%), was {before}; unsure held {unsure_held}/{}",
        his.len(),
        100.0 * false_holds as f64 / his.len() as f64,
        not_his.len(),
        100.0 * caught as f64 / not_his.len() as f64,
        unsure.len()
    );
    assert_eq!((his.len(), not_his.len()), (773, 95));
    assert!(false_holds <= 11, "his speech held {false_holds}");
    assert!(caught >= 89, "caught {caught}");
}

#[test]
fn his_profile_threshold_sits_on_the_plateau_of_his_data() {
    let rows = rows();
    let his = rows.iter().filter(|r| r.label == 'H').count() as f64;
    let not_his = rows.iter().filter(|r| r.label == 'F').count() as f64;
    let j = |bar: f32| {
        let fh = rows
            .iter()
            .filter(|r| r.label == 'H' && held_at(r, bar))
            .count() as f64;
        let ca = rows
            .iter()
            .filter(|r| r.label == 'F' && held_at(r, bar))
            .count() as f64;
        ca / not_his - fh / his
    };
    let (best_bar, best_j) = rows
        .iter()
        .map(|r| r.best)
        .map(|b| (b, j(b)))
        .fold((0.0, f64::MIN), |a, x| if x.1 > a.1 { x } else { a });
    println!(
        "Youden J at {BAR}: {:.3}; optimum {best_j:.3} at {best_bar:.3}",
        j(BAR)
    );
    assert!(best_j - j(BAR) < 0.03);
}

#[test]
fn the_reported_paste_is_held() {
    // 2026-10-18 17:24:58 +0545: film line, best window 0.37, ASR 0.24.
    let r = Row {
        best: 0.37,
        conf: Some(0.238),
        already_held: false,
        label: 'F',
    };
    assert!(held(&r));
    // 12:09:42Z Instagram reel: best 0.40, recognizer sure (0.82).
    let r = Row {
        best: 0.40,
        conf: Some(0.82),
        already_held: false,
        label: 'F',
    };
    assert!(held(&r));
}
