//! Names the training step learned — the machine-written half of the
//! dictionary. The user's own glossary (`~/.always/glossary.json`) stays the
//! hand-edited half; this file is written by Iris's training step from his
//! contacts, messages and Slack, with how often each name appears.
//!
//! File: `~/.always/vocab_names.json` (override: `ALWAYS_VOCAB_NAMES_PATH`).
//! Reloaded on mtime change, so a writer never restarts the daemon. Write it
//! atomically (temp file + rename).
//!
//! ```json
//! { "version": 1,
//!   "names": [
//!     { "term": "Livio", "frequency": 412, "source": "slack" },
//!     { "term": "Claude Code", "frequency": 97, "source": "messages",
//!       "mistranscriptions": ["cloud code"] } ] }
//! ```
//!
//! A bare JSON array of the same objects is accepted too. `frequency` is
//! the count of times the name appeared in his data; it orders Apple's
//! `contextualStrings` budget and acts as the matcher's prior (a name he
//! uses often may match a little more loosely). Entries longer than four
//! words / 40 chars are ignored (not vocabulary). At most
//! [`MAX_NAMES`] entries are used, highest frequency first.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::SystemTime;

/// Upper bound on names used (by frequency). Keeps the matcher's compiled
/// table and its memory bounded whatever the training step writes.
pub const MAX_NAMES: usize = 5000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearnedName {
    pub term: String,
    #[serde(default)]
    pub frequency: i64,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub mistranscriptions: Vec<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum NamesFile {
    Wrapped { names: Vec<LearnedName> },
    Bare(Vec<LearnedName>),
}

#[derive(Serialize)]
struct NamesFileOut<'a> {
    version: u32,
    names: &'a [LearnedName],
}

pub fn names_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ALWAYS_VOCAB_NAMES_PATH") {
        return Some(PathBuf::from(p));
    }
    dirs::home_dir().map(|h| h.join(".always").join("vocab_names.json"))
}

struct Cache {
    mtime: Option<SystemTime>,
    names: Vec<LearnedName>,
    loaded: bool,
}

static CACHE: OnceLock<RwLock<Cache>> = OnceLock::new();

fn cache() -> &'static RwLock<Cache> {
    CACHE.get_or_init(|| {
        RwLock::new(Cache {
            mtime: None,
            names: Vec::new(),
            loaded: false,
        })
    })
}

fn file_mtime() -> Option<SystemTime> {
    names_path()
        .and_then(|p| std::fs::metadata(p).ok())
        .and_then(|m| m.modified().ok())
}

/// Parse a names file body: valid entries only, deduped by lowercase
/// term (highest frequency kept), sorted frequency-desc, capped.
pub fn parse(raw: &str) -> anyhow::Result<Vec<LearnedName>> {
    let file: NamesFile = serde_json::from_str(raw)?;
    let names = match file {
        NamesFile::Wrapped { names } | NamesFile::Bare(names) => names,
    };
    let mut best: std::collections::HashMap<String, LearnedName> = Default::default();
    for mut n in names {
        n.term = n.term.trim().to_string();
        if !crate::glossary::is_vocab_sized(&n.term) {
            continue;
        }
        let key = n.term.to_lowercase();
        match best.get(&key) {
            Some(prev) if prev.frequency >= n.frequency => {}
            _ => {
                best.insert(key, n);
            }
        }
    }
    let mut out: Vec<LearnedName> = best.into_values().collect();
    out.sort_by(|a, b| b.frequency.cmp(&a.frequency).then(a.term.cmp(&b.term)));
    out.truncate(MAX_NAMES);
    Ok(out)
}

/// Current learned names (reloaded when the file's mtime changed).
pub fn names() -> Vec<LearnedName> {
    let mtime = file_mtime();
    {
        let c = cache().read();
        if c.loaded && c.mtime == mtime {
            return c.names.clone();
        }
    }
    let names = match names_path().map(std::fs::read_to_string) {
        Some(Ok(raw)) => parse(&raw).unwrap_or_else(|e| {
            tracing::warn!(error = %e, "vocab_names_parse_failed");
            Vec::new()
        }),
        _ => Vec::new(),
    };
    let mut c = cache().write();
    c.mtime = mtime;
    c.names = names.clone();
    c.loaded = true;
    names
}

/// The file's mtime as last seen — a version key for derived caches.
pub fn loaded_version() -> Option<SystemTime> {
    let _ = names();
    cache().read().mtime
}

/// Write `names` atomically to the names file (temp + rename). For the
/// daemon's own callers and tests; the training step may write the file
/// directly in the same format.
pub fn write(names: &[LearnedName]) -> anyhow::Result<()> {
    let path = names_path().ok_or_else(|| anyhow::anyhow!("no home directory"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    let body = serde_json::to_string_pretty(&NamesFileOut { version: 1, names })?;
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_wrapped_and_bare_forms_and_drops_sentences() {
        let wrapped = r#"{"version":1,"names":[
            {"term":"Livio","frequency":5,"source":"slack"},
            {"term":" livio ","frequency":9},
            {"term":"this is a whole sentence not a name","frequency":99},
            {"term":"Claude Code","frequency":3,"mistranscriptions":["cloud code"]}]}"#;
        let n = parse(wrapped).unwrap();
        assert_eq!(n.len(), 2);
        assert_eq!(n[0].term, "livio");
        assert_eq!(n[0].frequency, 9);
        assert_eq!(n[1].mistranscriptions, vec!["cloud code".to_string()]);

        let bare = r#"[{"term":"Jev","frequency":2}]"#;
        assert_eq!(parse(bare).unwrap()[0].term, "Jev");
    }
}
