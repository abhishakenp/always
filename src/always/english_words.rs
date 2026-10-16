//! "Is this an ordinary English word?" — the context gate of the dictionary
//! matcher (`vocab_match`) needs it.
//!
//! A recogniser that does not know a name writes either a non-word
//! ("Libio", "LDV") or real words ("leave you", "views"). Rewriting a
//! non-word toward a sound-alike dictionary term is safe. Rewriting real
//! words is not: "I'll leave you alone" must stay. So spans made only of
//! real words need context evidence before they may become a term.
//!
//! Source: the system word list (`/usr/share/dict/words`, 236k entries on
//! macOS), held as one sorted lowercase blob + offsets (~3.4 MB, loaded
//! once, off the paste path via [`warm`]). The list has no inflections,
//! so `-s/-es/-ies/-ed/-ing/-er/-ly` are stripped before lookup, and no
//! contractions, so the part before an apostrophe is checked.
//!
//! If the list is missing, every word counts as English: the matcher then
//! only rewrites with context evidence — fail-safe, never more rewrites.

use std::sync::OnceLock;

struct WordList {
    blob: Vec<u8>,
    /// (start, len) of each word in `blob`, sorted by the word bytes.
    spans: Vec<(u32, u16)>,
}

impl WordList {
    fn contains(&self, w: &[u8]) -> bool {
        self.spans
            .binary_search_by(|&(s, l)| self.blob[s as usize..s as usize + l as usize].cmp(w))
            .is_ok()
    }
}

static LIST: OnceLock<Option<WordList>> = OnceLock::new();

const SYSTEM_LIST: &str = "/usr/share/dict/words";

fn load(path: &str) -> Option<WordList> {
    let raw = std::fs::read(path).ok()?;
    let mut blob: Vec<u8> = raw.to_ascii_lowercase();
    // Normalise line endings to '\n'.
    blob.retain(|&b| b != b'\r');
    let mut spans: Vec<(u32, u16)> = Vec::new();
    let mut start = 0usize;
    for (i, &b) in blob.iter().enumerate() {
        if b == b'\n' {
            if i > start && i - start < u16::MAX as usize {
                spans.push((start as u32, (i - start) as u16));
            }
            start = i + 1;
        }
    }
    if start < blob.len() {
        spans.push((start as u32, (blob.len() - start) as u16));
    }
    let mut list = WordList { blob, spans };
    let mut spans = std::mem::take(&mut list.spans);
    spans.sort_by(|a, b| {
        let wa = &list.blob[a.0 as usize..a.0 as usize + a.1 as usize];
        let wb = &list.blob[b.0 as usize..b.0 as usize + b.1 as usize];
        wa.cmp(wb)
    });
    spans.dedup_by(|a, b| {
        list.blob[a.0 as usize..a.0 as usize + a.1 as usize]
            == list.blob[b.0 as usize..b.0 as usize + b.1 as usize]
    });
    list.spans = spans;
    if list.spans.is_empty() {
        return None;
    }
    Some(list)
}

fn list() -> Option<&'static WordList> {
    LIST.get_or_init(|| {
        let started = std::time::Instant::now();
        let l = load(SYSTEM_LIST);
        tracing::info!(
            words = l.as_ref().map(|l| l.spans.len()).unwrap_or(0),
            load_ms = started.elapsed().as_millis() as u64,
            "english_word_list_loaded"
        );
        l
    })
    .as_ref()
}

/// Load the list now (daemon start), so the first utterance never pays it.
pub fn warm() {
    let _ = list();
}

/// Words the system list lacks but speech is full of.
const EXTRA: &[&str] = &[
    "i", "im", "ive", "ill", "id", "youre", "youve", "youll", "youd", "hes", "shes", "its", "were",
    "theyre", "theyve", "dont", "doesnt", "didnt", "cant", "wont", "isnt", "arent", "wasnt",
    "werent", "okay", "ok", "yeah", "yep", "nope", "hey", "hi", "hello", "thanks", "gonna",
    "wanna", "gotta", "email", "online", "app", "apps", "website", "internet",
];

/// True when `token` (any case, may carry punctuation) is an ordinary
/// English word or an inflection of one. Empty / non-alphabetic tokens
/// return true: they are never evidence of a name.
pub fn is_english_word(token: &str) -> bool {
    let core: String = token
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '\'' || *c == '’')
        .collect::<String>()
        .to_lowercase()
        .replace('’', "'");
    if core.is_empty() || !core.chars().any(|c| c.is_ascii_alphabetic()) {
        return true;
    }
    let flat: String = core.chars().filter(|c| *c != '\'').collect();
    if EXTRA.contains(&flat.as_str()) {
        return true;
    }
    let Some(list) = list() else {
        return true;
    };
    // A contraction is English whatever its stem ("ain't", "y'all're");
    // a possessive is judged by its stem ("Livio's" → "livio").
    if let Some((_, after)) = core.rsplit_once('\'')
        && matches!(after, "t" | "ll" | "re" | "ve" | "d" | "m")
    {
        return true;
    }
    let base = core.split('\'').next().unwrap_or(&core);
    if base.is_empty() {
        return true;
    }
    if !base.bytes().all(|b| b.is_ascii_alphabetic()) {
        // Digits or non-ASCII letters: not a dictionary word, but not
        // evidence of a misheard English name either.
        return !base.chars().any(|c| c.is_ascii_alphabetic());
    }
    is_listed(list, base)
}

fn is_listed(list: &WordList, w: &str) -> bool {
    let b = w.as_bytes();
    if list.contains(b) {
        return true;
    }
    let try_stem = |stem: &str| stem.len() >= 2 && list.contains(stem.as_bytes());
    let strip = |suffix: &str| w.strip_suffix(suffix);
    if let Some(s) = strip("ies")
        && try_stem(&format!("{s}y"))
    {
        return true;
    }
    for suffix in ["s", "es", "ed", "d", "ing", "er", "ers", "est", "ly", "'s"] {
        if let Some(s) = strip(suffix) {
            if try_stem(s) {
                return true;
            }
            // "making" → "make", "used" → "use" handled by "d".
            if (suffix == "ing" || suffix == "ed" || suffix == "er") && try_stem(&format!("{s}e")) {
                return true;
            }
            // Doubled consonant: "running" → "run".
            if (suffix == "ing" || suffix == "ed")
                && s.len() >= 3
                && s.as_bytes()[s.len() - 1] == s.as_bytes()[s.len() - 2]
                && try_stem(&s[..s.len() - 1])
            {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_words_and_inflections_are_english() {
        if list().is_none() {
            return; // no system list on this host: everything is English
        }
        for w in [
            "leave", "you", "views", "video", "alone", "I'll", "don't", "running", "checked",
            "Give", "code", "cloud", "clock", "making", "stories", "ain't",
        ] {
            assert!(is_english_word(w), "{w} should be English");
        }
    }

    #[test]
    fn misheard_names_are_not_english() {
        if list().is_none() {
            return;
        }
        for w in [
            "Livio", "Libio", "Leevio", "LDV", "USPR", "Livio's", "Livios",
        ] {
            assert!(!is_english_word(w), "{w} should not be English");
        }
    }
}
