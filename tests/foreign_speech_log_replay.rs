//! Replays the user's REAL daemon logs through the foreign-speech gate
//! (`hallucination::foreign_speech`) and checks it against hand labels.
//!
//! The logs (`~/Library/Logs/Always/always.<date>`) are private and never
//! copied into the repo: the test reads them in place and skips when they
//! are absent (CI, other machines).
//!
//! Evidence is reconstructed exactly as the daemon would build it:
//! `streaming_final_used` → live session heard words, `streaming_final_empty`
//! → live session finished with no words; a chunked utterance never carries
//! a live-session signal (the current code drops the session for chunked
//! utterances). Acoustic confidence was not logged before this change, so
//! the confidence rule is NOT exercised here — see the unit tests in
//! `hallucination.rs` for that (measured on real audio).
//!
//! Labels (assigned by reading every pasted transcript of 2026-10-11..14):
//! - 2026-10-11 17:03–18:28 UTC: an 80-minute family conversation in
//!   Nepali/Hindi that the gate let through as ~250 garbled English pastes
//!   ("Rings of Nanjani, Baina.", "Nasu Nati", …) → misfire.
//! - 2026-10-14 03:45:05: "And a baggy?" (the reported bug) → misfire.
//! - a few short non-English words outside that window ("Alaykum", "Aja",
//!   "Care", "I like Babu") → suspected misfire, reported separately.
//! - 2026-10-12 12:39:38 "No": a 4.5 s utterance that became one word,
//!   between other speakers → ambiguous, reported separately.
//! - everything else (real dictation) → genuine. `[audio saved: …]`
//!   placeholders are not STT output and are skipped.

use always::always::hallucination::foreign_speech;
use always::stt::{LiveStreamOutcome, RecognitionEvidence, TranscriptionResult};

const DAYS: &[&str] = &["2026-10-11", "2026-10-12", "2026-10-13", "2026-10-14"];
const STORM: (&str, &str) = ("2026-10-11T17:03:00", "2026-10-11T18:28:00");
const BUG: &str = "2026-10-14T03:45:05";
const SUSPECT: &[&str] = &[
    "2026-10-12T09:31:17",
    "2026-10-13T08:27:16",
    "2026-10-13T08:27:41",
    "2026-10-13T08:28:10",
];
const AMBIGUOUS: &[&str] = &["2026-10-12T12:39:38"];
/// Labels cover pastes up to the moment they were assigned; later pastes
/// in a still-growing log are unlabelled and skipped.
const LABELLED_UNTIL: &str = "2026-10-14T03:49:00";

#[derive(Debug, Clone, Copy, PartialEq)]
enum Label {
    Genuine,
    Misfire,
    Suspect,
    Ambiguous,
}

struct Utterance {
    ts: String,
    text: String,
    live: LiveStreamOutcome,
    label: Label,
}

fn label(ts: &str) -> Label {
    let sec = &ts[..19];
    if sec == BUG || (sec >= STORM.0 && sec < STORM.1) {
        Label::Misfire
    } else if SUSPECT.contains(&sec) {
        Label::Suspect
    } else if AMBIGUOUS.contains(&sec) {
        Label::Ambiguous
    } else {
        Label::Genuine
    }
}

fn load() -> Option<Vec<Utterance>> {
    let dir = dirs::home_dir()?.join("Library/Logs/Always");
    let mut out = Vec::new();
    for day in DAYS {
        let Ok(body) = std::fs::read_to_string(dir.join(format!("always.{day}"))) else {
            return None;
        };
        let mut live = LiveStreamOutcome::NotRun;
        let mut chunked = false;
        for line in body.lines() {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let f = &v["fields"];
            match f["message"].as_str().unwrap_or("") {
                "voice_detected" => {
                    live = LiveStreamOutcome::NotRun;
                    chunked = false;
                }
                "streaming_final_used" => live = LiveStreamOutcome::Heard,
                "streaming_final_empty" => live = LiveStreamOutcome::NoWords,
                "streaming_final_skipped_chunked" => chunked = true,
                "chunked_utterance_assembled" => chunked = true,
                "transcription_pasted" => {
                    let ts = v["timestamp"].as_str().unwrap_or("").to_string();
                    let text = f["raw_text"].as_str().unwrap_or("").to_string();
                    if !text.starts_with("[audio saved")
                        && ts.len() >= 19
                        && ts[..19] < *LABELLED_UNTIL
                    {
                        out.push(Utterance {
                            label: label(&ts),
                            ts,
                            text,
                            live: if chunked {
                                LiveStreamOutcome::NotRun
                            } else {
                                live
                            },
                        });
                    }
                    live = LiveStreamOutcome::NotRun;
                    chunked = false;
                }
                _ => {}
            }
        }
    }
    Some(out)
}

fn rejected(u: &Utterance) -> bool {
    let r = TranscriptionResult {
        text: u.text.clone(),
        evidence: RecognitionEvidence {
            live_stream: u.live,
            ..Default::default()
        },
        ..Default::default()
    };
    foreign_speech(&r).is_some()
}

#[test]
fn replay_real_logs_through_foreign_speech_gate() {
    let Some(utts) = load() else {
        eprintln!("SKIP: ~/Library/Logs/Always/always.2026-10-1[1-4] not present");
        return;
    };
    let count = |l: Label| utts.iter().filter(|u| u.label == l).count();
    let rej = |l: Label| utts.iter().filter(|u| u.label == l && rejected(u)).count();
    let misfire_live_empty = utts
        .iter()
        .filter(|u| u.label == Label::Misfire && u.live == LiveStreamOutcome::NoWords)
        .count();

    eprintln!("pasted utterances replayed: {}", utts.len());
    eprintln!(
        "genuine   passed   {}/{}",
        count(Label::Genuine) - rej(Label::Genuine),
        count(Label::Genuine)
    );
    eprintln!(
        "misfires  rejected {}/{}  (of which live-session-empty: {}/{})",
        rej(Label::Misfire),
        count(Label::Misfire),
        rej(Label::Misfire),
        misfire_live_empty
    );
    eprintln!(
        "suspects  rejected {}/{}",
        rej(Label::Suspect),
        count(Label::Suspect)
    );
    eprintln!(
        "ambiguous rejected {}/{}",
        rej(Label::Ambiguous),
        count(Label::Ambiguous)
    );
    for u in utts
        .iter()
        .filter(|u| u.label != Label::Misfire && rejected(u))
    {
        eprintln!(
            "  rejected non-misfire: {} {:?} {:?}",
            u.ts, u.label, u.text
        );
    }

    // Zero regressions on genuine dictation.
    let bad: Vec<_> = utts
        .iter()
        .filter(|u| u.label == Label::Genuine && rejected(u))
        .map(|u| format!("{} {:?}", u.ts, u.text))
        .collect();
    assert!(bad.is_empty(), "genuine dictation rejected: {bad:#?}");

    // The reported bug is rejected.
    let bug: Vec<_> = utts.iter().filter(|u| u.ts.starts_with(BUG)).collect();
    assert_eq!(bug.len(), 1);
    assert_eq!(bug[0].text, "And a baggy?");
    assert!(rejected(bug[0]), "\"And a baggy?\" must be rejected");

    // Every misfire whose live session heard nothing is rejected.
    assert_eq!(rej(Label::Misfire), misfire_live_empty);
}
