//! Replay of the 2026-10-14 dictation fragmentation through the
//! end-of-utterance decision (`classify_pause_text` +
//! `final_silence_cut_frames`).
//!
//! Source: ~/Library/Logs/Always/always.2026-10-14, 10:55-11:25Z. With
//! `auto_enter_delay_ms = 0`, every utterance cut is one paste and one
//! Return, so "cuts" below = Enters. Pieces are the recogniser's raw text
//! (tails as logged) and the pause after each piece:
//! - intra-utterance pauses are the logged `speech_resumed_after_pause`
//!   values (30-600 ms);
//! - fragment boundaries are upper bounds measured from the log
//!   (`voice_detected` of the next fragment − `preview_asr_utterance_end`
//!   of this one + the 0.6 s window that cut it).

use super::{
    FRAME_MS, MIDSENTENCE_DECISION_GRACE_FRAMES, PauseText, PauseVerdict, classify_pause_text,
    endpoint_cut_frames, extended_silence_frames, final_silence_cut_frames,
    hesitation_silence_frames, partial_says_hesitating,
};
use crate::always::AlwaysConfig;
use crate::always::localization::Localization;

/// Prod: `stt_silence = 0.6` → 20 frames of 30 ms.
const BASE: usize = 20;

fn hesitation() -> usize {
    hesitation_silence_frames(&AlwaysConfig::default())
}

/// How the loop decides one pause, frame by frame: the deciding text
/// lands `text_latency_ms` into the pause (speculative STT or the flushed
/// chunk's raw STT). Returns `true` if the pause ends the utterance before
/// speech resumes at `pause_ms` (`None` = never resumes).
fn pause_cuts(text: &str, pause_ms: Option<u32>, text_latency_ms: u32) -> bool {
    pause_cut_ms(text, pause_ms, text_latency_ms).is_some()
}

/// Silence (ms) at which the pause ends the utterance, `None` if speech
/// resumes first.
fn pause_cut_ms(text: &str, pause_ms: Option<u32>, text_latency_ms: u32) -> Option<u32> {
    let loc = &Localization::ENGLISH;
    let pause_frames = pause_ms.map(|ms| (ms / FRAME_MS) as usize);
    let latency_frames = (text_latency_ms / FRAME_MS) as usize;
    for frame in 1..=10_000usize {
        if pause_frames.is_some_and(|p| frame > p) {
            return None; // speech resumed first: same utterance
        }
        let decided = frame >= latency_frames;
        let cut = endpoint_cut_frames(
            true,
            BASE,
            BASE,
            hesitation(),
            PauseVerdict {
                text_pending: !decided,
                text: decided.then(|| classify_pause_text(loc, text)),
                partial_hesitating: false,
            },
        );
        if frame >= cut {
            return Some(frame as u32 * FRAME_MS);
        }
    }
    panic!("a pause that never resumes must end the utterance");
}

/// Replays one spoken sequence in paste mode; returns the number of
/// Enters. His auto-Enter setting (0) is authoritative since 83ef87a — no
/// commit hold, no merge — so every cut is one paste and one Return.
fn enters_for(pieces: &[(&str, Option<u32>)], text_latency_ms: u32) -> usize {
    pieces
        .iter()
        .filter(|(text, pause)| pause_cut_ms(text, *pause, text_latency_ms).is_some())
        .count()
}

/// The same sequence under prod 83ef87a (paste mode cut at the base
/// window whatever the text said): every pause of at least the window.
fn enters_prod(pieces: &[(&str, Option<u32>)]) -> usize {
    pieces
        .iter()
        .filter(|(_, pause)| pause.is_none_or(|p| p >= BASE as u32 * FRAME_MS))
        .count()
}

/// Logged Apple STT for a 6-10 s chunk landed 152-551 ms after the cut
/// (`stt_wait_ms`), i.e. inside the hold.
const TEXT_LATENCY_MS: u32 = 450;

#[test]
fn hesitation_hold_default_is_1_2_s() {
    assert_eq!(hesitation(), 40); // ceil(1200 / 30)
    assert_eq!(hesitation(), extended_silence_frames(BASE));
}

#[test]
fn today_fragment_endings_classify() {
    let loc = &Localization::ENGLISH;
    // Fragments that were pasted+Entered while the user kept talking,
    // and that end on positive hesitation evidence.
    for t in [
        "I think we should show the, uh, my uh, transcription in the website that goes all the...",
        "not disturb anything on the screen would just show in the upside and after like a tiny...",
        "Just, uh, go away, uh, it would just uh, collapse back and if I want to.",
        "but it would be really nice to...",
        "Maybe, uh, and, uh, I think it's that, uh,",
        "or the entire had no delay, but it was still so great and.",
        "it's not even clear what, you know",
        "it was, like",
        "so we should, um",
    ] {
        assert_eq!(classify_pause_text(loc, t), PauseText::Hesitating, "{t}");
    }
    // Complete sentences: unchanged, finish at the base window.
    for t in [
        "What time is it?",
        "Send the email now.",
        "It is a pretty good idea. Basically, it would feel the whole lip screen as need.",
        "I don't know.",
    ] {
        assert_eq!(classify_pause_text(loc, t), PauseText::Complete, "{t}");
    }
    assert_eq!(
        classify_pause_text(loc, "Very nice to have"),
        PauseText::MidSentence
    );
    // Partials (no punctuation): only a positive hesitation counts.
    assert!(partial_says_hesitating(loc, "show it on the left and uh"));
    assert!(partial_says_hesitating(
        loc,
        "it should go all the way to the"
    ));
    assert!(!partial_says_hesitating(loc, "what time is it"));
}

/// 11:04:32-11:04:47Z: "...after like a tiny..." → 1.21 s → "Just, uh, go
/// away, ... if I want to." → 1.44 s → "I could just uh go to the
/// detailed view uh in the notch and everything." Logged: 3 pastes, 3
/// Enters (each 51 ms after its paste). Both long pauses outlast the 1.2 s
/// hesitation hold, so they still cut.
#[test]
fn replay_1104_tiny_go_away_detailed_view() {
    let pieces: &[(&str, Option<u32>)] = &[
        (
            "Basically, because it would not disturb anything on",
            Some(540),
        ),
        (
            "the screen would just show in the upside and after like a tiny...",
            Some(1_210),
        ),
        ("Just, uh, go away,", Some(300)),
        (
            "uh, it would just uh, collapse back and if I want to.",
            Some(1_440),
        ),
        ("I could just uh go to the detailed view uh", Some(90)),
        ("in the notch and everything.", None),
    ];
    // (paste mode now, prod 83ef87a). No commit hold merges a resume any
    // more: a pause past the hold is a Return.
    assert_eq!(
        (enters_for(pieces, TEXT_LATENCY_MS), enters_prod(pieces)),
        (3, 3)
    );
}

/// 11:09:01-11:09:11Z: "...but it would be really nice to..." → 1.63 s →
/// "The transcript, uh, to the right, the strip, ..." Logged: 2 Enters.
#[test]
fn replay_1109_really_nice_to() {
    let pieces: &[(&str, Option<u32>)] = &[
        (
            "And basically, Iris spoke, and after she was done speaking,",
            Some(480),
        ),
        (
            "I saw the transcript at the bottom, but it would be really nice to...",
            Some(1_630),
        ),
        ("The transcript, uh, to the right, the strip,", Some(30)),
        ("uh, not the bottom, the right strip,", Some(660)),
        ("and as she was speaking, not, uh, after.", None),
    ];
    // (paste mode now, prod 83ef87a). No commit hold merges a resume any
    // more: a pause past the hold is a Return.
    assert_eq!(
        (enters_for(pieces, TEXT_LATENCY_MS), enters_prod(pieces)),
        (2, 3)
    );
}

/// 11:04:10-11:04:17Z: "...that goes all the..." → ≤2.43 s → "To the left
/// as needed, ..." Logged: 2 Enters.
#[test]
fn replay_1104_goes_all_the() {
    let pieces: &[(&str, Option<u32>)] = &[
        (
            "I think we should show the, uh, my uh, transcription in the website that goes all the...",
            Some(2_430),
        ),
        (
            "To the left as needed, and uh, the Irish uh, transcript,",
            Some(30),
        ),
        ("uh, it should go all the way to it.", None),
    ];
    // (paste mode now, prod 83ef87a). No commit hold merges a resume any
    // more: a pause past the hold is a Return.
    assert_eq!(
        (enters_for(pieces, TEXT_LATENCY_MS), enters_prod(pieces)),
        (2, 2)
    );
}

/// The live partial hears "uh" before any final text exists: the hold
/// applies from the first check, even if the chunk's STT is slow.
#[test]
fn partial_hesitation_holds_before_text_lands() {
    let v = PauseVerdict {
        text_pending: true,
        text: None,
        partial_hesitating: true,
    };
    assert_eq!(
        final_silence_cut_frames(BASE, BASE, hesitation(), v),
        hesitation()
    );
}

/// Intra-utterance pauses of 30-600 ms (all logged values today) never
/// cut, whatever the text says.
#[test]
fn short_pauses_never_cut() {
    for ms in [
        30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 420, 450, 480, 540, 570,
    ] {
        assert!(!pause_cuts("Send the email now.", Some(ms), 0), "{ms} ms");
        assert!(!pause_cuts("and uh", Some(ms), 0), "{ms} ms");
    }
}

/// A complete short command still ends at the base window — no added
/// delay: cut at frame 20 (600 ms) once its text is in, same as before.
#[test]
fn complete_command_enters_at_base_window() {
    let loc = &Localization::ENGLISH;
    let decided = |t: &str| PauseVerdict {
        text_pending: false,
        text: Some(classify_pause_text(loc, t)),
        partial_hesitating: false,
    };
    for t in ["What time is it?", "Send the email now.", "Stop!"] {
        assert_eq!(
            final_silence_cut_frames(BASE, BASE, hesitation(), decided(t)),
            BASE,
            "{t}"
        );
    }
    // Text still in flight: the pre-existing decision grace, unchanged.
    let pending = PauseVerdict {
        text_pending: true,
        text: None,
        partial_hesitating: false,
    };
    assert_eq!(
        final_silence_cut_frames(BASE, BASE, hesitation(), pending),
        BASE + MIDSENTENCE_DECISION_GRACE_FRAMES
    );
    // No text at all (adaptive off / short utterance): plain base window.
    let none = PauseVerdict {
        text_pending: false,
        text: None,
        partial_hesitating: false,
    };
    assert_eq!(
        final_silence_cut_frames(BASE, BASE, hesitation(), none),
        BASE
    );
    assert_eq!(enters_for(&[("What time is it?", None)], 0), 1);
    assert!(pause_cuts("What time is it?", Some(10_000), 0));
}

/// Never hangs: a hesitation followed by real silence (the 10.4 s gap
/// at 11:17:29Z) still ends the utterance — at the hesitation window.
#[test]
fn hesitation_then_long_silence_still_ends() {
    assert!(pause_cuts(
        "Maybe, uh, and, uh, I think it's that, uh,",
        Some(10_400),
        TEXT_LATENCY_MS
    ));
    assert!(pause_cuts("goes all the...", None, TEXT_LATENCY_MS));
}

/// Honest boundary: a fragment that ends as a COMPLETE sentence still
/// ends at the base window even if the user resumes 1.0-1.8 s later —
/// e.g. 11:03:38Z "...the whole lip screen as need." → ≤1.07 s. No text
/// says "more is coming" there, and the user ruled out a blanket delay.
#[test]
fn complete_sentence_then_resume_still_cuts() {
    assert!(pause_cuts(
        "It is a pretty good idea. Basically, it would feel the whole lip screen as need.",
        Some(1_070),
        TEXT_LATENCY_MS
    ));
}

/// 2026-10-15 18:53:48Z: a stale live partial ("…uh") set the hesitation
/// hold at frame 28; the flushed chunk's text "…We have to send a PR."
/// decided Complete at frame 70, yet the cut waited for frame 84 (2.5 s
/// after he stopped). Decided Complete text now ends the pause at once.
#[test]
fn decided_complete_text_overrides_stale_partial_hesitation() {
    let loc = &Localization::ENGLISH;
    let text = "check out to the main branch and do the work and everything and also after \
                making changes and installing locally and testing and everything. We have to send a PR.";
    let v = PauseVerdict {
        text_pending: false,
        text: Some(classify_pause_text(loc, text)),
        partial_hesitating: true,
    };
    assert_eq!(v.text, Some(PauseText::Complete));
    assert_eq!(final_silence_cut_frames(BASE, BASE, hesitation(), v), BASE);
    // Text still in flight: the partial's hold stands (nothing better yet).
    let pending = PauseVerdict {
        text_pending: true,
        text: None,
        partial_hesitating: true,
    };
    assert_eq!(
        final_silence_cut_frames(BASE, BASE, hesitation(), pending),
        hesitation()
    );
    // Decided text that is itself unfinished keeps the partial's hold.
    let mid = PauseVerdict {
        text_pending: false,
        text: Some(classify_pause_text(loc, "and then we")),
        partial_hesitating: true,
    };
    assert_eq!(
        final_silence_cut_frames(BASE, BASE, hesitation(), mid),
        extended_silence_frames(BASE).max(hesitation())
    );
    // The decided text itself trailing off on a filler still holds.
    let hes = PauseVerdict {
        text_pending: false,
        text: Some(classify_pause_text(loc, "I think it's that, uh,")),
        partial_hesitating: false,
    };
    assert_eq!(
        final_silence_cut_frames(BASE, BASE, hesitation(), hes),
        extended_silence_frames(BASE).max(hesitation())
    );
}

/// 2026-10-15 21:49:31Z and 21:49:46Z (0.1.0): decided text ending on a
/// connector ("…that you need to", "…what's happening, or.") held the cut
/// 2.5 s — `endpoint_hold_ms` 2517 / 2518. Now the hold is 1.2 s.
#[test]
fn trailing_connector_cuts_at_1_2_s() {
    let loc = &Localization::ENGLISH;
    for t in [
        "Everything that you need to",
        "seems broken in always or what, what's happening, or.",
    ] {
        let v = PauseVerdict {
            text_pending: false,
            text: Some(classify_pause_text(loc, t)),
            partial_hesitating: false,
        };
        let cut = final_silence_cut_frames(BASE, BASE, hesitation(), v) as u32 * FRAME_MS;
        assert!(cut <= 1_200, "{t}: {cut} ms");
    }
    // He resumes 2.43 s into the pause (the 10-14 upper bound): cut at
    // 1.2 s, resume 1.2 s after the paste < 3 s hold → one message.
    let pieces: &[(&str, Option<u32>)] = &[
        ("Everything that you need to", Some(2_430)),
        ("check is in the log.", None),
    ];
    // (paste mode now, prod 83ef87a). No commit hold merges a resume any
    // more: a pause past the hold is a Return.
    assert_eq!(
        (enters_for(pieces, TEXT_LATENCY_MS), enters_prod(pieces)),
        (2, 2)
    );
}

/// 2026-10-15 18:56:06Z: "…and this is about all." cut at 0.72 s and was
/// Entered; he resumed 0.7 s after the paste. Text that reads finished
/// gives no evidence to hold on — the honest boundary.
#[test]
fn replay_185606_complete_looking_thinking_pause() {
    let pieces: &[(&str, Option<u32>)] = &[
        (
            "This, I'm not sure if this is the exact code you have to check, test, and fix, and this is about all.",
            Some(1_420),
        ),
        ("Takes a while to do that and while I was speaking.", None),
    ];
    // (paste mode now, prod 83ef87a). No commit hold merges a resume any
    // more: a pause past the hold is a Return.
    assert_eq!(
        (enters_for(pieces, TEXT_LATENCY_MS), enters_prod(pieces)),
        (2, 2)
    );
}

/// Paste mode (auto-Enter 0, no commit hold): positive "more is coming"
/// evidence holds the cut exactly as in consume mode; only text still in
/// flight no longer adds the decision grace (the paste does not wait on
/// that text). Complete / no text: the base window, no added delay.
#[test]
fn paste_mode_holds_open_clauses_but_never_waits_for_text() {
    let loc = &Localization::ENGLISH;
    let held = extended_silence_frames(BASE).max(hesitation());
    let v = |pending: bool, text: Option<&str>, partial: bool| PauseVerdict {
        text_pending: pending,
        text: text.map(|t| classify_pause_text(loc, t)),
        partial_hesitating: partial,
    };
    // Open clause: held in both modes.
    for open in [
        v(true, None, true),
        v(false, Some("Everything that you need to"), false),
        v(false, Some("and then we"), false),
        v(false, Some("Great limited one account so."), false),
    ] {
        assert_eq!(endpoint_cut_frames(true, BASE, BASE, hesitation(), open), held, "{open:?}");
        assert_eq!(endpoint_cut_frames(false, BASE, BASE, hesitation(), open), held, "{open:?}");
    }
    // Text in flight: no grace when pasting; consume mode keeps it.
    let pending = v(true, None, false);
    assert_eq!(endpoint_cut_frames(true, BASE, BASE, hesitation(), pending), BASE);
    assert_eq!(
        endpoint_cut_frames(false, BASE, BASE, hesitation(), pending),
        BASE + MIDSENTENCE_DECISION_GRACE_FRAMES
    );
    // Finished: base window in both.
    for done in [
        v(false, Some("What's the status on that?"), false),
        v(false, Some("Please fix that."), false),
        v(false, None, false),
    ] {
        assert_eq!(endpoint_cut_frames(true, BASE, BASE, hesitation(), done), BASE, "{done:?}");
    }
}

/// His decided pause texts that ENDED the utterance read as finished now;
/// open clauses still hold (2026-10-12..16 logs, `endpoint_log_replay`).
#[test]
fn connector_before_terminator_reads_finished_unless_a_conjunction() {
    let loc = &Localization::ENGLISH;
    for t in [
        "I just tried, it did not work. What's happening? What's the status on that?",
        "Okay, so is the click lag being worked on?",
        "Anything like that?",
        "Please fix that.",
        "And disturbed me from watching movies and that is why I said that.",
        "I should otherwise ask if it's irreversible action or things like that.",
    ] {
        assert_eq!(classify_pause_text(loc, t), PauseText::Complete, "{t}");
    }
    for t in [
        "Great limited one account so.",
        "It can happen in the.",
        "Everything that you need to.",
        "seems broken in always or what, what's happening, or.",
        "And?",
        "Check the documentation of Omnir out and everything that...",
    ] {
        assert_eq!(classify_pause_text(loc, t), PauseText::Hesitating, "{t}");
    }
    // Live partials (unpunctuated): "that" is no evidence; "the" still is.
    assert!(!partial_says_hesitating(loc, "whats the status on that"));
    assert!(partial_says_hesitating(loc, "it should go all the way to the"));
}

/// Landed speculation replaces the live session's finish only when the
/// session heard words and the speculation is valid with text.
#[test]
fn speculation_replaces_stream_only_with_full_evidence() {
    use super::speculation_replaces_stream as r;
    assert!(r(true, true, true, true));
    assert!(!r(false, true, true, true)); // no session: speculation path anyway
    assert!(!r(true, false, true, true)); // invalidated by a resume
    assert!(!r(true, true, false, true)); // not landed / empty
    assert!(!r(true, true, true, false)); // session heard nothing: let it report NoWords
}

/// One decided pause from his live logs (see `endpoint_log_replay`).
struct LoggedPause {
    text: String,
    decided_ms: u32,
    /// `Some(ms)` = he resumed after this much (smoothed) silence.
    resumed_ms: Option<u32>,
}

/// Silence (ms) at which `cut_frames(verdict_known)` ends this pause, or
/// `None` when he resumed first. The verdict is known from `decided_ms`.
fn logged_cut_ms(p: &LoggedPause, cut_frames: impl Fn(Option<PauseText>) -> usize) -> Option<u32> {
    let loc = &Localization::ENGLISH;
    let verdict = classify_pause_text(loc, &p.text);
    for frame in 1..=10_000u32 {
        let ms = frame * FRAME_MS;
        if p.resumed_ms.is_some_and(|r| ms > r) {
            return None;
        }
        let known = (ms >= p.decided_ms).then_some(verdict);
        if frame as usize >= cut_frames(known) {
            return Some(ms);
        }
    }
    None
}

/// Replays every decided pause of his pasted utterances (extracted from
/// ~/Library/Logs/Always by the scratch `extract.py` into JSONL with
/// `text`, `frame` (decision frame), `outcome` ("end"/"resumed"),
/// `pause_ms`) through the paste-mode endpoint. Prints, per policy, the
/// thinking pauses it would cut (a cut = paste + Return at auto-Enter 0)
/// and the endpoint hold on finished utterances.
#[test]
#[ignore = "needs ENDPOINT_LOG_REPLAY=<decisions.jsonl>; run explicitly"]
fn endpoint_log_replay() {
    let Ok(path) = std::env::var("ENDPOINT_LOG_REPLAY") else {
        return;
    };
    let pauses: Vec<LoggedPause> = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter_map(|v| {
            Some(LoggedPause {
                text: v["text"].as_str()?.to_string(),
                decided_ms: v["frame"].as_u64()? as u32 * FRAME_MS,
                resumed_ms: match v["outcome"].as_str()? {
                    "resumed" => Some(v["pause_ms"].as_u64()? as u32),
                    _ => None,
                },
            })
        })
        .collect();
    let base = BASE;
    let held = extended_silence_frames(BASE).max(hesitation());
    let policies: Vec<(&str, Box<dyn Fn(Option<PauseText>) -> usize>)> = vec![
        ("prod 83ef87a: base always", Box::new(move |_| base)),
        (
            "new: paste-mode holds",
            Box::new(move |v| {
                endpoint_cut_frames(
                    true,
                    base,
                    base,
                    hesitation(),
                    PauseVerdict {
                        text_pending: v.is_none(),
                        text: v,
                        partial_hesitating: false,
                    },
                )
            }),
        ),
        (
            "new + hesitation hold 1.8 s",
            Box::new(move |v| {
                endpoint_cut_frames(
                    true,
                    base,
                    base,
                    60,
                    PauseVerdict {
                        text_pending: false,
                        text: v,
                        partial_hesitating: false,
                    },
                )
            }),
        ),
        (
            "new + hesitation hold 2.5 s",
            Box::new(move |v| {
                endpoint_cut_frames(
                    true,
                    base,
                    base,
                    84,
                    PauseVerdict {
                        text_pending: false,
                        text: v,
                        partial_hesitating: false,
                    },
                )
            }),
        ),
        (
            "fast: Complete cuts at 210 ms",
            Box::new(move |v| match v {
                Some(PauseText::Complete) => 7,
                Some(_) => held,
                None => base,
            }),
        ),
        (
            "fast: Complete cuts at 450 ms",
            Box::new(move |v| match v {
                Some(PauseText::Complete) => 15,
                Some(_) => held,
                None => base,
            }),
        ),
    ];
    let resumed = pauses.iter().filter(|p| p.resumed_ms.is_some()).count();
    let ended = pauses.len() - resumed;
    // Learned end-of-turn (`eot.rs`): the logs keep no audio, so the
    // model's score on these pauses is unknown. What IS known: every
    // resumed pause here lasted ≥ its logged (smoothed) silence + the
    // 300 ms Silero hold in raw frames, i.e. past every probe (≤ 570 ms),
    // so the model could reach all of them; the live-partial veto
    // (`partial_says_hesitating` on the same words, unpunctuated) blocks
    // some. The rest are bounded by the model's held-out false-cut rate
    // on his audio pause rows (`eot_corpus::eot_replay`).
    {
        let loc = &Localization::ENGLISH;
        let partial = |t: &str| -> String {
            t.chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '\'')
                .collect::<String>()
                .to_lowercase()
        };
        let reach: Vec<&LoggedPause> = pauses
            .iter()
            .filter(|p| p.resumed_ms.is_some_and(|ms| ms + 300 > 6 * FRAME_MS))
            .collect();
        let vetoed = reach
            .iter()
            .filter(|p| partial_says_hesitating(loc, &partial(&p.text)))
            .count();
        eprintln!(
            "learned EOT: resumed pauses reachable by a probe {}/{resumed}, partial-veto {vetoed}, exposed {} (added splits = exposed x held-out false-cut rate)",
            reach.len(),
            reach.len() - vetoed
        );
    }
    eprintln!("pauses: {} ({resumed} resumed thinking pauses, {ended} ends)", pauses.len());
    for (name, cut) in &policies {
        let splits = pauses
            .iter()
            .filter(|p| p.resumed_ms.is_some() && logged_cut_ms(p, cut).is_some())
            .count();
        let mut holds: Vec<u32> = pauses
            .iter()
            .filter(|p| p.resumed_ms.is_none())
            .filter_map(|p| logged_cut_ms(p, cut))
            .collect();
        holds.sort_unstable();
        // Speculation landed by the cut: the paste skips the live
        // session's finish (`speculation_replaces_stream`).
        let landed = pauses
            .iter()
            .filter(|p| p.resumed_ms.is_none())
            .filter(|p| logged_cut_ms(p, cut).is_some_and(|c| p.decided_ms <= c))
            .count();
        let pct = |q: f64| holds[((holds.len() as f64 - 1.0) * q) as usize];
        eprintln!(
            "{name:30} splits {splits:3}/{resumed}  end hold p50 {} p90 {} p95 {} mean {} ms  text landed by cut {landed}/{ended}",
            pct(0.5),
            pct(0.9),
            pct(0.95),
            holds.iter().sum::<u32>() / holds.len() as u32,
        );
        if std::env::var("ENDPOINT_LOG_REPLAY_VERBOSE").is_ok() {
            for p in pauses.iter().filter(|p| p.resumed_ms.is_some()) {
                if let Some(c) = logged_cut_ms(p, cut) {
                    eprintln!("    split at {c} ms (resumed {:?}): {:?}", p.resumed_ms, p.text);
                }
            }
        }
    }
}
