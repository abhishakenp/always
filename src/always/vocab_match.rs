//! The dictionary matcher: makes words the user added (or the training
//! step learned) come out of every speech engine spelled right.
//!
//! Why it exists (2026-10-16, his log 04:42:48Z): he added `Livio` with the
//! wrong forms the recogniser had produced before, then said "Livio" alone.
//! Apple's file pass heard "Leave you.", the live stream "Give you" — real
//! English words, none of them a listed wrong form — and a two-word
//! utterance skipped the correction stack entirely. Recognisers bias toward
//! a term only weakly (`contextualStrings`), so the text has to be fixed
//! after recognition, on every path that can produce pasted text.
//!
//! Per utterance, left to right, for each 1–4 word span:
//!
//! 1. **Learned wrong forms** (glossary `mistranscriptions`, names-file
//!    `mistranscriptions`) match exactly and rewrite deterministically —
//!    except a form made only of ordinary English words that shares no word
//!    with its term ("leave you" for Livio): that needs the context gate
//!    below, or "I'll leave you alone" would break. A single curated
//!    common word ("cloud") is left to the existing tiered matcher, which
//!    owns its context rules.
//! 2. **Phonetic match** (`vocab_phonetic`): the span's sound vs every
//!    term's sound, weighted edit distance over phoneme classes.
//!    - A span holding a non-word ("Libio", "Leevio's") may become a term
//!      at a tight threshold — a recogniser only writes a non-word when it
//!      is guessing.
//!    - A span sharing a word with a multi-word term ("cloud code" /
//!      `Claude Code`, "Livios PR" / `Livio's PR`) is anchored by it.
//!    - A span of only real words ("leave you") needs context evidence: it
//!      is the whole utterance ("Leave you."), or it fills a name slot
//!      ("thanks, leave you.", "ask leave you."). Never one bare word.
//!
//! The term's recent use (pasted in the last 15 min) and its frequency in
//! his data loosen the threshold slightly — a prior, never a trigger.
//!
//! Cost: tokens are classified once; only spans that can qualify are
//! scored, against terms bucketed by first sound and filtered by length.
//! Measured in `tests::latency_*` (target ≤ 2 ms per utterance). No LLM.

use crate::always::english_words::is_english_word;
use crate::always::vocab_phonetic::{self as ph, Ph};
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant, SystemTime};

/// Phonetic score at or below which a qualified span becomes a term.
pub const MATCH_THRESHOLD: f32 = 0.12;
/// Looser bound when the span shares a word with a multi-word term.
pub const ANCHORED_THRESHOLD: f32 = 0.16;
/// Extra allowance when the term was pasted recently.
const RECENCY_BONUS: f32 = 0.03;
/// Extra allowance for a term frequent in his data.
const FREQUENCY_BONUS: f32 = 0.02;
const FREQUENT: i64 = 500;
const RECENT_WINDOW: Duration = Duration::from_secs(15 * 60);
const MAX_SPAN: usize = 4;

/// Utterance words that carry no content for the "whole utterance" test.
const FILLERS: &[&str] = &[
    "um", "uh", "oh", "ok", "okay", "so", "and", "hey", "hi", "hello", "yeah", "yes", "please",
    "thanks", "well", "like",
];
/// Hesitation sounds: never part of a name ("uh very" is not Avery).
const HESITATIONS: &[&str] = &["uh", "um", "er", "ah", "eh", "hmm", "mm"];
/// A name usually follows these when it ends the clause.
const NAME_CUES: &[&str] = &[
    "hey", "hi", "hello", "thanks", "ping", "cc", "dear", "ask", "tell", "message", "email", "dm",
    "call", "with", "from",
];

#[derive(Debug, Clone)]
struct Term {
    canonical: String,
    /// Lowercase alphanumeric tokens of the canonical form.
    tokens: Vec<String>,
    phon: Vec<Ph>,
    frequency: i64,
    /// Stated by the user (glossary) or among his most frequent learned
    /// names. Untrusted names (the long tail of a contacts import) only
    /// match by sound inside a name slot — with 1308 names loaded, sound
    /// alone turned "Alexa" into Alex and "mini" into Manny.
    trusted: bool,
}

#[derive(Debug, Default)]
pub struct Compiled {
    terms: Vec<Term>,
    /// First-sound bucket → term indexes.
    buckets: HashMap<u8, Vec<usize>>,
    /// Normalised wrong form (tokens joined by ' ') → (term, form tokens).
    exact: HashMap<String, (usize, Vec<String>)>,
    /// Every word of a multi-word term: only a span holding one can be
    /// anchored, so other all-English spans skip scoring entirely.
    anchor_words: std::collections::HashSet<String>,
    /// Non-words from his own history (see [`ACCEPTED_MIN_COUNT`]).
    accepted: std::collections::HashSet<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HitKind {
    Exact,
    Phonetic { score: f32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub from: String,
    pub to: String,
    pub kind: HitKind,
}

/// One term source for [`Compiled::build`].
pub struct TermInput {
    pub term: String,
    pub mistranscriptions: Vec<String>,
    pub frequency: i64,
    pub trusted: bool,
}

/// Learned names trusted like the user's own terms: the most frequent ones.
pub const TRUSTED_NAMES: usize = 50;
/// Untrusted names need a tighter sound match on top of a name slot.
const UNTRUSTED_THRESHOLD: f32 = 0.04;
/// A non-word he has dictated this many times without correcting it is his
/// own word ("Alexa": 26 times in transcripts.jsonl), never a mishearing.
const ACCEPTED_MIN_COUNT: usize = 3;

fn norm_token(w: &str) -> String {
    w.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn tokens_of(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(norm_token)
        .filter(|t| !t.is_empty())
        .collect()
}

/// First-sound bucket: voicing pairs and vowels share one, so "Libio" and
/// "Pivio" land with Livio's neighbours but "give you" does not.
fn bucket(p: Ph) -> u8 {
    match p {
        b'a' | b'e' | b'i' | b'o' | b'u' => b'a',
        b'B' | b'P' | b'V' | b'F' => b'B',
        b'T' | b'D' | b'Q' => b'T',
        b'K' | b'G' => b'K',
        b'S' | b'Z' | b'X' | b'C' | b'J' => b'S',
        b'M' | b'N' => b'M',
        b'L' | b'R' => b'L',
        b'W' | b'Y' | b'H' => b'W',
        other => other,
    }
}

impl Compiled {
    pub fn build(inputs: impl IntoIterator<Item = TermInput>) -> Self {
        let mut c = Compiled::default();
        let mut seen: HashMap<String, usize> = HashMap::new();
        for input in inputs {
            let canonical = input.term.trim().to_string();
            if !crate::glossary::is_vocab_sized(&canonical) {
                continue;
            }
            let key = canonical.to_lowercase();
            let idx = match seen.get(&key) {
                Some(&i) => {
                    c.terms[i].frequency = c.terms[i].frequency.max(input.frequency);
                    c.terms[i].trusted |= input.trusted;
                    i
                }
                None => {
                    let tokens = tokens_of(&canonical);
                    if tokens.is_empty() {
                        continue;
                    }
                    let phon = ph::encode_span(canonical.split_whitespace());
                    if phon.is_empty() {
                        continue;
                    }
                    let i = c.terms.len();
                    c.buckets.entry(bucket(phon[0])).or_default().push(i);
                    if tokens.len() > 1 {
                        c.anchor_words.extend(tokens.iter().cloned());
                    }
                    c.terms.push(Term {
                        canonical,
                        tokens,
                        phon,
                        frequency: input.frequency,
                        trusted: input.trusted,
                    });
                    seen.insert(key, i);
                    i
                }
            };
            for wrong in &input.mistranscriptions {
                let toks = tokens_of(wrong);
                if toks.is_empty() || toks.len() > MAX_SPAN || toks == c.terms[idx].tokens {
                    continue;
                }
                c.exact.entry(toks.join(" ")).or_insert((idx, toks));
            }
        }
        c
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Learn his accepted non-words from past transcripts: tokens outside
    /// the English list that recur ≥ [`ACCEPTED_MIN_COUNT`] times, minus
    /// any learned wrong form (those he did correct).
    pub fn accept_history<'a>(&mut self, texts: impl IntoIterator<Item = &'a str>) {
        let words = recurring_non_words(texts);
        self.accept_words(&words);
    }

    fn accept_words(&mut self, words: &std::collections::HashSet<String>) {
        let wrong: std::collections::HashSet<&str> = self
            .exact
            .values()
            .flat_map(|(_, f)| f.iter().map(String::as_str))
            .collect();
        self.accepted = words
            .iter()
            .filter(|w| !wrong.contains(w.as_str()))
            .cloned()
            .collect();
    }
}

fn recurring_non_words<'a>(
    texts: impl IntoIterator<Item = &'a str>,
) -> std::collections::HashSet<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for t in texts {
        for w in t.split_whitespace() {
            let norm = norm_token(split_punct(w).1);
            if norm.len() >= 3 && !is_english_word(w) {
                *counts.entry(norm).or_default() += 1;
            }
        }
    }
    counts
        .into_iter()
        .filter(|(_, n)| *n >= ACCEPTED_MIN_COUNT)
        .map(|(w, _)| w)
        .collect()
}

/// His accepted non-words, read from history once per process (at the
/// daemon's warm-up thread, never on a paste).
static ACCEPTED: OnceLock<std::collections::HashSet<String>> = OnceLock::new();

fn accepted_words() -> &'static std::collections::HashSet<String> {
    ACCEPTED.get_or_init(|| {
        let history = history_texts();
        recurring_non_words(history.iter().map(String::as_str))
    })
}

/// Texts of `~/.always/transcripts.jsonl` (his pasted history).
fn history_texts() -> Vec<String> {
    let Some(path) = crate::always::transcript_stream::stream_path() else {
        return Vec::new();
    };
    let Ok(raw) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    raw.lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter_map(|v| v.get("text").and_then(|t| t.as_str()).map(str::to_string))
        .collect()
}

/// Recently pasted text, for the recency prior.
static RECENT: OnceLock<Mutex<VecDeque<(Instant, String)>>> = OnceLock::new();

fn recent() -> &'static Mutex<VecDeque<(Instant, String)>> {
    RECENT.get_or_init(|| Mutex::new(VecDeque::new()))
}

/// Record pasted text (call once per paste).
pub fn note_pasted(text: &str) {
    let mut r = recent().lock();
    let now = Instant::now();
    while r
        .front()
        .is_some_and(|(t, _)| now.duration_since(*t) > RECENT_WINDOW)
    {
        r.pop_front();
    }
    if r.len() >= 64 {
        r.pop_front();
    }
    r.push_back((now, text.to_lowercase()));
}

fn recently_used(term_lower: &str) -> bool {
    let r = recent().lock();
    let now = Instant::now();
    r.iter()
        .any(|(t, s)| now.duration_since(*t) <= RECENT_WINDOW && s.contains(term_lower))
}

/// Context the gate reads besides the text itself.
pub trait Prior {
    /// Was the term (canonical form) pasted recently?
    fn recently_used(&self, canonical: &str) -> bool;
}

struct LivePrior;
impl Prior for LivePrior {
    fn recently_used(&self, canonical: &str) -> bool {
        recently_used(&canonical.to_lowercase())
    }
}

/// No recency (tests, measurement).
pub struct NoPrior;
impl Prior for NoPrior {
    fn recently_used(&self, _: &str) -> bool {
        false
    }
}

struct Tok<'a> {
    raw: &'a str,
    norm: String,
    english: bool,
    /// Code-like (digits, paths, identifiers): never a spoken name.
    codeish: bool,
    phon: Vec<Ph>,
    /// Raw token ends a clause (`.`, `,`, `!`, `?`, `;`, `:`).
    clause_end: bool,
}

fn split_punct(w: &str) -> (&str, &str, &str) {
    let start = w
        .char_indices()
        .find(|(_, c)| c.is_alphanumeric())
        .map(|(i, _)| i)
        .unwrap_or(w.len());
    let end = w
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_alphanumeric())
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(start);
    (&w[..start], &w[start..end.max(start)], &w[end.max(start)..])
}

fn classify(raw: &str) -> Tok<'_> {
    let (_, core, tail) = split_punct(raw);
    let norm = norm_token(core);
    let codeish = core
        .chars()
        .any(|c| c.is_ascii_digit() || matches!(c, '/' | '_' | '.' | '=' | '\\'))
        || !core.is_ascii();
    Tok {
        raw,
        english: is_english_word(core),
        phon: if codeish {
            Vec::new()
        } else {
            ph::encode_word(core)
        },
        norm,
        codeish,
        clause_end: tail.contains(['.', ',', '!', '?', ';', ':']),
    }
}

/// Correct `text` against the live dictionary (glossary + learned names).
/// Returns the text unchanged when nothing qualifies.
pub fn correct(text: &str) -> String {
    correct_hits(text).0
}

/// [`correct`] plus the rewrites made, logged once per rewrite.
pub fn correct_hits(text: &str) -> (String, Vec<Hit>) {
    let compiled = compiled();
    let (out, hits) = correct_with(text, &compiled, &LivePrior);
    for h in &hits {
        crate::glossary::bump_frequency(&h.to);
        if crate::always::telemetry::should_log_transcripts() {
            tracing::info!(from = %h.from, to = %h.to, kind = ?h.kind, "vocab_match_applied");
        } else {
            tracing::info!(to = %h.to, kind = ?h.kind, "vocab_match_applied");
        }
    }
    (out, hits)
}

/// [`correct`] without logging or frequency bumps — for live previews,
/// which re-run on every partial.
pub fn correct_quiet(text: &str) -> String {
    correct_with(text, &compiled(), &LivePrior).0
}

/// Testable core: no globals.
pub fn correct_with(text: &str, c: &Compiled, prior: &dyn Prior) -> (String, Vec<Hit>) {
    if c.is_empty() || text.trim().is_empty() {
        return (text.to_string(), Vec::new());
    }
    let raws: Vec<&str> = text.split_whitespace().collect();
    let toks: Vec<Tok> = raws.iter().map(|r| classify(r)).collect();
    // Content tokens for the "whole utterance" test.
    let content: Vec<usize> = toks
        .iter()
        .enumerate()
        .filter(|(_, t)| !t.norm.is_empty() && !FILLERS.contains(&t.norm.as_str()))
        .map(|(i, _)| i)
        .collect();

    let mut out: Vec<String> = Vec::with_capacity(raws.len());
    let mut hits = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        match best_at(i, &toks, &content, c, prior) {
            Some((n, term_idx, kind)) => {
                let (prefix, _, _) = split_punct(toks[i].raw);
                let (_, _, suffix) = split_punct(toks[i + n - 1].raw);
                let canonical = &c.terms[term_idx].canonical;
                let from = raws[i..i + n].join(" ");
                out.push(format!("{prefix}{canonical}{suffix}"));
                hits.push(Hit {
                    from,
                    to: canonical.clone(),
                    kind,
                });
                i += n;
            }
            None => {
                out.push(raws[i].to_string());
                i += 1;
            }
        }
    }
    if hits.is_empty() {
        return (text.to_string(), hits);
    }
    (out.join(" "), hits)
}

/// The best rewrite starting at token `i`: (span length, term, kind).
fn best_at(
    i: usize,
    toks: &[Tok],
    content: &[usize],
    c: &Compiled,
    prior: &dyn Prior,
) -> Option<(usize, usize, HitKind)> {
    let max_n = MAX_SPAN.min(toks.len() - i);
    // A span that is the whole utterance's content ("Leave you.").
    let isolated = |n: usize| -> bool {
        content.len() <= MAX_SPAN
            && !content.is_empty()
            && content.iter().all(|&k| k >= i && k < i + n)
            && (i..i + n).all(|k| content.contains(&k))
    };
    // A span after a name cue that ends its clause ("thanks, leave you.").
    let name_slot = |n: usize| -> bool {
        let prev_cue = i > 0 && NAME_CUES.contains(&toks[i - 1].norm.as_str());
        let ends = i + n == toks.len() || toks[i + n - 1].clause_end;
        prev_cue && ends
    };

    // 1. Learned wrong forms, longest first.
    for n in (1..=max_n).rev() {
        let span = &toks[i..i + n];
        // Never join across a sentence/clause mark ("…not views. PR and").
        if span.iter().any(|t| t.norm.is_empty()) || span[..n - 1].iter().any(|t| t.clause_end) {
            continue;
        }
        let key = span
            .iter()
            .map(|t| t.norm.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let Some((term_idx, form)) = c.exact.get(&key) else {
            continue;
        };
        let term = &c.terms[*term_idx];
        let all_english = span.iter().all(|t| t.english);
        let shares_word = form.iter().any(|f| term.tokens.contains(f));
        if all_english && n == 1 && crate::always::common_words::is_common_word(&key) {
            // "cloud" → Claude: the tiered matcher's context rules own it.
            continue;
        }
        if all_english && !shares_word && !(isolated(n) || name_slot(n)) {
            continue;
        }
        return Some((n, *term_idx, HitKind::Exact));
    }

    // 2. Phonetic.
    let mut best: Option<(usize, usize, f32)> = None;
    for n in (1..=max_n).rev() {
        let span = &toks[i..i + n];
        if span
            .iter()
            .any(|t| t.norm.is_empty() || t.codeish || HESITATIONS.contains(&t.norm.as_str()))
        {
            continue;
        }
        // Punctuation inside the span splits clauses: never join across it.
        if span[..n - 1].iter().any(|t| t.clause_end) {
            continue;
        }
        let all_english = span.iter().all(|t| t.english);
        // A non-word he says all the time is his word, not a mishearing.
        if span
            .iter()
            .any(|t| !t.english && c.accepted.contains(&t.norm))
        {
            continue;
        }
        let context = isolated(n) || name_slot(n);
        // Qualify the span before touching any term.
        let may_anchor = n >= 2 && span.iter().any(|t| c.anchor_words.contains(&t.norm));
        // A non-word may carry at most one real word along ("Lee Vio",
        // "Libio's"): "at the Libio" is two words plus a name.
        let english_count = span.iter().filter(|t| t.english).count();
        let by_sound = if all_english {
            n >= 2 && context
        } else {
            english_count <= 1
        };
        if !may_anchor && !by_sound {
            continue;
        }
        let phon: Vec<Ph> = span.iter().flat_map(|t| t.phon.iter().copied()).collect();
        if phon.len() < 3 {
            continue;
        }
        let Some(cands) = c.buckets.get(&bucket(phon[0])) else {
            continue;
        };
        for &ti in cands {
            let term = &c.terms[ti];
            if n > term.tokens.len() + 1 {
                continue;
            }
            let slack = 2.max(term.phon.len() / 3);
            if phon.len().abs_diff(term.phon.len()) > slack {
                continue;
            }
            // Anchor: same word count, a shared word in the same place,
            // and at least one word that differs.
            let anchored = term.tokens.len() > 1
                && n == term.tokens.len()
                && span.iter().zip(&term.tokens).any(|(t, w)| &t.norm == w)
                && span.iter().zip(&term.tokens).any(|(t, w)| &t.norm != w);
            let qualified = if anchored {
                true
            } else if !all_english {
                // A lone short non-word is too collision-prone; a name
                // from the long tail also needs a name slot.
                !(n == 1 && phon.len() < 4) && (term.trusted || context)
            } else {
                n >= 2 && context && term.trusted
            };
            if !qualified {
                continue;
            }
            let mut threshold = if anchored {
                ANCHORED_THRESHOLD
            } else if term.trusted {
                MATCH_THRESHOLD
            } else {
                UNTRUSTED_THRESHOLD
            };
            if prior.recently_used(&term.canonical) {
                threshold += RECENCY_BONUS;
            }
            if term.frequency >= FREQUENT {
                threshold += FREQUENCY_BONUS;
            }
            // Length alone already costs ≥ 0.4 per missing phoneme.
            let len = phon.len().max(term.phon.len()) as f32;
            if phon.len().abs_diff(term.phon.len()) as f32 * 0.4 / len > threshold {
                continue;
            }
            let s = ph::score_within(&phon, &term.phon, threshold);
            if s > threshold {
                continue;
            }
            // Already the term's words. Only a lost apostrophe or hyphen
            // ("Livios PR") is worth a rewrite; case alone is left as said.
            let same = span.len() == term.tokens.len()
                && span.iter().zip(&term.tokens).all(|(t, w)| &t.norm == w);
            if same {
                let said = span
                    .iter()
                    .map(|t| split_punct(t.raw).1.to_lowercase())
                    .collect::<Vec<_>>()
                    .join(" ");
                if said == term.canonical.to_lowercase() {
                    return None;
                }
                return Some((n, ti, HitKind::Exact));
            }
            // Lower score wins; ties keep the longer span (seen first).
            if best.is_none_or(|(_, _, bs)| s < bs) {
                best = Some((n, ti, s));
            }
        }
    }
    best.map(|(n, ti, s)| (n, ti, HitKind::Phonetic { score: s }))
}

/// Is `wrong → right` a dictionary correction worth learning without
/// review? `right` must be word-sized and either hold a non-word (a name:
/// "Livio") or sound like `wrong` ("cloud code" → "Claude Code"). A
/// rewrite of ordinary words into other ordinary words ("great" → "good")
/// is an edit, not vocabulary.
pub fn is_vocab_pair(wrong: &str, right: &str) -> bool {
    if !crate::glossary::is_vocab_sized(right) || !crate::glossary::is_vocab_sized(wrong) {
        return false;
    }
    if tokens_of(wrong) == tokens_of(right) {
        return false;
    }
    let names_something = right.split_whitespace().any(|w| !is_english_word(w));
    let a = ph::encode_span(wrong.split_whitespace());
    let b = ph::encode_span(right.split_whitespace());
    names_something || ph::score(&a, &b) <= 0.35
}

/// `always vocab eval`: replay the matcher (live dictionary, no recency
/// prior) over every transcript in `files` and report each rewrite plus the
/// per-utterance cost. Lines are JSON: a `text` field (transcripts.jsonl) or
/// a daemon log line with `fields.raw_text` (`transcription_pasted`).
/// Exact duplicates are replayed once.
pub fn eval_files(files: &[String]) -> anyhow::Result<String> {
    use std::fmt::Write as _;
    let mut report = String::new();
    crate::always::english_words::warm();
    let c = compiled();
    let mut seen = std::collections::HashSet::new();
    let mut texts: Vec<String> = Vec::new();
    for f in files {
        let raw = std::fs::read_to_string(f)?;
        for line in raw.lines() {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let text = v
                .get("text")
                .and_then(|t| t.as_str())
                .or_else(|| v.pointer("/fields/raw_text").and_then(|t| t.as_str()));
            if let Some(t) = text
                && !t.trim().is_empty()
                && seen.insert(t.to_string())
            {
                texts.push(t.to_string());
            }
        }
    }
    let mut times: Vec<Duration> = Vec::with_capacity(texts.len());
    let mut rewritten = 0usize;
    for t in &texts {
        let started = Instant::now();
        let (out, hits) = correct_with(t, &c, &NoPrior);
        times.push(started.elapsed());
        if !hits.is_empty() {
            rewritten += 1;
            let _ = writeln!(report, "- {t}\n+ {out}");
            for h in &hits {
                let _ = writeln!(report, "    {:?}: {:?} -> {:?}", h.kind, h.from, h.to);
            }
        }
    }
    times.sort();
    let pct = |p: f64| {
        times
            .get(((times.len() as f64 * p) as usize).min(times.len().saturating_sub(1)))
            .copied()
            .unwrap_or_default()
    };
    let _ = writeln!(
        report,
        "\n{} unique transcripts, {} rewritten; dictionary {} terms / {} wrong forms; per utterance p50 {:?} p99 {:?} max {:?}",
        texts.len(),
        rewritten,
        c.terms.len(),
        c.exact.len(),
        pct(0.5),
        pct(0.99),
        times.last().copied().unwrap_or_default()
    );
    Ok(report)
}

// ---------------------------------------------------------------------------
// Live dictionary
// ---------------------------------------------------------------------------

type Version = (Option<SystemTime>, Option<SystemTime>);
static COMPILED: OnceLock<Mutex<(Option<Version>, Arc<Compiled>)>> = OnceLock::new();

/// The dictionary compiled from the glossary's user-stated terms and the
/// training step's names; rebuilt when either file changes.
pub fn compiled() -> Arc<Compiled> {
    let version = (
        crate::glossary::loaded_version(),
        crate::always::vocab_names::loaded_version(),
    );
    let cell = COMPILED.get_or_init(|| Mutex::new((None, Arc::new(Compiled::default()))));
    let mut g = cell.lock();
    if g.0 != Some(version) {
        let glossary = crate::glossary::vocab_entries()
            .into_iter()
            .filter(|e| e.user_stated)
            .map(|e| TermInput {
                term: e.term,
                mistranscriptions: e.mistranscriptions,
                frequency: e.frequency,
                trusted: true,
            });
        // `names()` is frequency-desc: the head is trusted.
        let names = crate::always::vocab_names::names()
            .into_iter()
            .enumerate()
            .map(|(rank, n)| TermInput {
                term: n.term,
                mistranscriptions: n.mistranscriptions,
                frequency: n.frequency,
                trusted: rank < TRUSTED_NAMES,
            });
        let mut built = Compiled::build(glossary.chain(names));
        built.accept_words(accepted_words());
        tracing::info!(
            terms = built.terms.len(),
            wrong_forms = built.exact.len(),
            accepted_words = built.accepted.len(),
            "vocab_match_compiled"
        );
        *g = (Some(version), Arc::new(built));
    }
    Arc::clone(&g.1)
}

/// Terms for engine-side biasing (Apple `contextualStrings`): the user's
/// stated glossary terms first, then learned names by frequency.
pub fn bias_terms(limit: usize) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    let stated = crate::glossary::vocab_entries()
        .into_iter()
        .filter(|e| e.user_stated)
        .map(|e| e.term);
    let names = crate::always::vocab_names::names()
        .into_iter()
        .map(|n| n.term);
    for t in stated.chain(names) {
        if out.len() >= limit {
            break;
        }
        if seen.insert(t.to_lowercase()) {
            out.push(t);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn his_dictionary() -> Compiled {
        let t = |term: &str, miss: &[&str]| TermInput {
            term: term.into(),
            mistranscriptions: miss.iter().map(|s| s.to_string()).collect(),
            frequency: 100,
            trusted: true,
        };
        Compiled::build(vec![
            t("Livio", &["LDV", "Leevio", "Libio"]),
            t("Livio's PR", &["USPR", "views PR", "Livios PR"]),
            t("Claude Code", &["clock code", "cloud code"]),
            t("Claude", &["cloud"]),
            t("winnow", &["Vino"]),
        ])
    }

    fn fix(text: &str) -> String {
        correct_with(text, &his_dictionary(), &NoPrior).0
    }

    #[test]
    fn his_livio_cases() {
        assert_eq!(fix("Leave you."), "Livio.");
        assert_eq!(fix("leave you"), "Livio");
        assert_eq!(fix("Hey, leave you."), "Hey, Livio.");
        assert_eq!(fix("Thanks leave you."), "Thanks Livio.");
        assert_eq!(fix("LDV"), "Livio");
        assert_eq!(
            fix("Libio, can you check this?"),
            "Livio, can you check this?"
        );
        assert_eq!(fix("ask Leevio about it"), "ask Livio about it");
    }

    #[test]
    fn his_pr_cases() {
        assert_eq!(fix("check views PR"), "check Livio's PR");
        assert_eq!(fix("Check USPR."), "Check Livio's PR.");
        assert_eq!(fix("check Livios PR please"), "check Livio's PR please");
        assert_eq!(fix("check Libio's PR"), "check Livio's PR");
        assert_eq!(fix("check Livio's PR"), "check Livio's PR");
    }

    #[test]
    fn his_claude_code_cases() {
        assert_eq!(fix("open cloud code"), "open Claude Code");
        assert_eq!(fix("Run it in clock code."), "Run it in Claude Code.");
        assert_eq!(
            fix("ask clawed code to fix it"),
            "ask Claude Code to fix it"
        );
    }

    #[test]
    fn ordinary_phrases_stay() {
        for s in [
            "I'll leave you alone",
            "leave video on",
            "I don't want to leave you.",
            "Give you",
            "give you a minute",
            "We should leave your laptop here.",
            "deploy this to the cloud",
            "Leave",
            "I said leave you, but she heard something else.",
            "views of the PR are fine",
            "Living in Lisbon.",
            // A wrong form split by a sentence mark is two sentences.
            "Livio's PR, not views. PR is fine.",
        ] {
            assert_eq!(fix(s), s, "{s} must stay unchanged");
        }
    }

    #[test]
    fn vocab_pairs_are_names_or_sound_alikes() {
        assert!(is_vocab_pair("Leave you", "Livio"));
        assert!(is_vocab_pair("cloud code", "Claude Code"));
        assert!(is_vocab_pair("views PR", "Livio's PR"));
        assert!(!is_vocab_pair("great", "good"));
        assert!(!is_vocab_pair(
            "I think we should",
            "we must certainly do this now"
        ));
        assert!(!is_vocab_pair("livio", "Livio"));
    }

    #[test]
    fn long_tail_names_need_a_name_slot() {
        let mut inputs: Vec<TermInput> = ["Alex", "Manny", "Kent", "Todd", "Morris", "Jason"]
            .iter()
            .map(|n| TermInput {
                term: n.to_string(),
                mistranscriptions: vec![],
                frequency: 3,
                trusted: false,
            })
            .collect();
        inputs.push(TermInput {
            term: "Livio".into(),
            mistranscriptions: vec![],
            frequency: 100,
            trusted: true,
        });
        let mut c = Compiled::build(inputs);
        c.accept_history(["Alexa what time is it", "Alexa stop", "hey Alexa"]);
        let fix = |s: &str| correct_with(s, &c, &NoPrior).0;
        assert_eq!(fix("Hey Alexa."), "Hey Alexa.");
        for s in [
            "Alexa, play music",
            "use the mini model",
            "to do.",
            "my Iris.",
            "parse the JSON",
        ] {
            assert_eq!(fix(s), s);
        }
        assert_eq!(fix("Leave you."), "Livio.");
        assert_eq!(fix("thanks Alekx."), "thanks Alex.");
    }

    #[test]
    fn idempotent() {
        for s in ["Leave you.", "check views PR", "open cloud code"] {
            let once = fix(s);
            assert_eq!(fix(&once), once);
        }
    }

    // Debug builds run 10-100x slower and flake under parallel tests;
    // check with `cargo test --release` (or `examples/vocab_eval.rs`).
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "latency budget holds for release builds only"
    )]
    fn latency_long_utterance_under_budget() {
        // 5000 learned names + his glossary, a 4000-char dictation.
        const A: &[&str] = &[
            "Li", "Ma", "San", "To", "Hi", "Ra", "Ke", "Pe", "Da", "Vi", "No", "Su", "Ga",
        ];
        const B: &[&str] = &[
            "vi", "rco", "ja", "bi", "ro", "ne", "lo", "ta", "mi", "ku", "sa",
        ];
        const C: &[&str] = &[
            "o", "a", "y", "as", "shi", "n", "el", "ia", "us", "ka", "ir", "e", "on", "et", "um",
            "ar", "is", "ou", "ai", "ek", "ol", "ez", "ad", "ub", "ox", "if", "ag", "ep", "il",
            "ot", "uz", "aj", "ev", "ow", "ix", "eb",
        ];
        let mut inputs: Vec<TermInput> = (0..5000)
            .map(|k| TermInput {
                term: format!(
                    "{}{}{}",
                    A[k % A.len()],
                    B[(k / A.len()) % B.len()],
                    C[(k / (A.len() * B.len())) % C.len()]
                ),
                mistranscriptions: vec![],
                frequency: (k % 700) as i64,
                trusted: k < TRUSTED_NAMES,
            })
            .collect();
        inputs.push(TermInput {
            term: "Livio".into(),
            mistranscriptions: vec!["LDV".into()],
            frequency: 100,
            trusted: true,
        });
        let c = Compiled::build(inputs);
        let sentence = "So I was looking at the Libio branch and the cloud code output, \
             then src/always/vad.rs line 3209 said leave you alone, okay? ";
        let text = sentence.repeat(4000 / sentence.len() + 1);
        let _ = correct_with(&text, &c, &NoPrior); // warm word list
        let mut worst = Duration::ZERO;
        for _ in 0..20 {
            let t = Instant::now();
            let _ = correct_with(&text, &c, &NoPrior);
            worst = worst.max(t.elapsed());
        }
        let short = Instant::now();
        let _ = correct_with("Leave you.", &c, &NoPrior);
        let short = short.elapsed();
        eprintln!("vocab_match latency: 4000-char worst {worst:?}, short {short:?}");
        // Debug builds are ~10x slower than the release daemon.
        assert!(worst < Duration::from_millis(2), "{worst:?}");
    }
}
