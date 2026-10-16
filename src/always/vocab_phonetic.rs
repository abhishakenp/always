//! Spelling → rough phoneme sequence, and a weighted edit distance over it.
//!
//! The dictionary matcher (`vocab_match`) compares what the recogniser
//! WROTE against what the user SAID. Recognisers write real English words
//! for a name they do not know: "Livio" comes back as "leave you", "Libio",
//! "LDV". Plain string distance cannot see that "leave you" and "Livio"
//! sound alike (5 of 8 letters differ), and Soundex keeps only consonants,
//! so "leave" alone would match too. This keeps vowels as classes:
//!
//!   leave you → L i V Y u        Livio → L i V i o
//!
//! and charges little for the substitutions a recogniser actually makes:
//! vowel ↔ vowel, the glide `Y` ↔ the vowel `i`, voicing pairs (B/V, D/T,
//! K/G, F/V, S/Z), a dropped vowel. A different consonant costs a full
//! point, so "give you" (G i V Y u) stays far from Livio.
//!
//! English spelling is irregular; these rules cover the common digraphs
//! and the silent final `e`, which is what the observed mishearings need.
//! No dictionary, no model — ~1 µs per word.

/// One phoneme class. Consonants are uppercase ASCII, vowels lowercase:
/// `a e i o u`. Special consonants: `X` = sh, `C` = ch, `J` = j/dge,
/// `Q` = th, `N` covers `ng`'s nasal.
pub type Ph = u8;

fn is_vowel_letter(c: u8) -> bool {
    matches!(c, b'a' | b'e' | b'i' | b'o' | b'u')
}

pub fn is_vowel(p: Ph) -> bool {
    p.is_ascii_lowercase()
}

/// Lowercase ASCII letters of `word`; everything else (apostrophes,
/// digits, punctuation) dropped. Non-ASCII letters are kept out: the
/// rules below are English-only and a non-English word simply encodes
/// to fewer symbols (and so matches nothing closely).
fn letters(word: &str) -> Vec<u8> {
    word.bytes()
        .filter(|b| b.is_ascii_alphabetic())
        .map(|b| b.to_ascii_lowercase())
        .collect()
}

/// Encode one written word.
pub fn encode_word(word: &str) -> Vec<Ph> {
    let mut w = letters(word);
    let mut out: Vec<Ph> = Vec::with_capacity(w.len());
    if w.is_empty() {
        return out;
    }
    // An all-consonant token of ≤ 4 letters is an acronym the recogniser
    // spelled out ("LDV", "PR"): letters are said by name, which these
    // spelling rules cannot model. Keep it literal so only an exact token
    // (or a learned wrong form) ever matches it.
    if w.len() <= 4 && !w.iter().any(|&c| is_vowel_letter(c) || c == b'y') {
        return w.iter().map(|c| c.to_ascii_uppercase()).collect();
    }
    // Silent final e: "leave", "code", "clawed"→ no, that ends in d.
    if w.len() > 2 && w[w.len() - 1] == b'e' && !is_vowel_letter(w[w.len() - 2]) {
        w.pop();
    }
    let n = w.len();
    let at = |i: usize| -> u8 { if i < n { w[i] } else { 0 } };
    let mut i = 0;
    while i < n {
        let c = w[i];
        let next = at(i + 1);
        // Vowel runs: digraphs first, then single vowels.
        if is_vowel_letter(c) {
            let pair = [c, next];
            let (ph, used) = match &pair {
                b"ee" | b"ea" | b"ie" | b"ei" => (b'i', 2),
                b"oo" | b"ou" | b"ue" => (b'u', 2),
                b"oa" | b"oe" => (b'o', 2),
                b"ai" => (b'a', 2),
                b"au" => (b'o', 2),
                _ => match (c, next) {
                    (b'e', b'w') => (b'u', 2),
                    (b'o', b'w') => (b'o', 2),
                    (b'a', b'w') | (b'a', b'y') => (if next == b'w' { b'o' } else { b'a' }, 2),
                    (b'e', b'y') => (b'i', 2),
                    _ => (c, 1),
                },
            };
            push(&mut out, ph);
            i += used;
            continue;
        }
        let (ph, used): (&[u8], usize) = match (c, next) {
            (b'p', b'h') => (b"F", 2),
            (b's', b'h') => (b"X", 2),
            (b't', b'c') if at(i + 2) == b'h' => (b"C", 3),
            (b'c', b'h') => (b"C", 2),
            (b't', b'h') => (b"Q", 2),
            (b'w', b'h') => (b"W", 2),
            (b'c', b'k') => (b"K", 2),
            (b'k', b'n') if i == 0 => (b"N", 2),
            (b'w', b'r') if i == 0 => (b"R", 2),
            (b'g', b'h') => {
                if i == 0 {
                    (b"G", 2)
                } else {
                    (b"", 2)
                }
            }
            (b'd', b'g') => (b"J", 2),
            (b'q', b'u') => (b"KW", 2),
            (b'n', b'g') => (b"N", 2),
            (b'x', _) => {
                if i == 0 {
                    (b"Z", 1)
                } else {
                    (b"KS", 1)
                }
            }
            (b'c', b'e' | b'i' | b'y') => (b"S", 1),
            (b'c', _) => (b"K", 1),
            (b'q', _) => (b"K", 1),
            (b'j', _) => (b"J", 1),
            // y: a consonant glide before a vowel at a syllable start,
            // otherwise the vowel i ("gym", "Livy").
            (b'y', _) => {
                if is_vowel_letter(next) && (i == 0 || !is_vowel_letter(w[i - 1])) {
                    (b"Y", 1)
                } else {
                    (b"i", 1)
                }
            }
            // w: a glide before a vowel, otherwise the vowel u.
            (b'w', _) => {
                if is_vowel_letter(next) {
                    (b"W", 1)
                } else {
                    (b"u", 1)
                }
            }
            _ => {
                let up = [c.to_ascii_uppercase()];
                // Doubled consonants say one sound ("Libbio", "Clauddde").
                if out.last() == Some(&up[0]) {
                    i += 1;
                    continue;
                }
                out.push(up[0]);
                i += 1;
                continue;
            }
        };
        for &p in ph {
            push(&mut out, p);
        }
        i += used;
    }
    out
}

/// Encode a run of capital letters as their spoken letter names, the way
/// a recogniser writes a name it heard as letters ("NVOPR" for "Livio
/// PR"). Letters whose name starts with a vowel sound (F L M N R S X)
/// keep only the consonant: in a fast run the vowel is not heard.
pub fn encode_letters(letters: &str) -> Vec<Ph> {
    let mut out = Vec::new();
    for c in letters.bytes().filter(u8::is_ascii_alphabetic) {
        let name: &[u8] = match c.to_ascii_uppercase() {
            b'A' => b"a",
            b'B' => b"Bi",
            b'C' => b"Si",
            b'D' => b"Di",
            b'E' => b"i",
            b'F' => b"F",
            b'G' => b"Ji",
            b'H' => b"C",
            b'I' => b"a",
            b'J' => b"Ja",
            b'K' => b"Ka",
            b'L' => b"L",
            b'M' => b"M",
            b'N' => b"N",
            b'O' => b"o",
            b'P' => b"Pi",
            b'Q' => b"KYu",
            b'R' => b"R",
            b'S' => b"S",
            b'T' => b"Ti",
            b'U' => b"Yu",
            b'V' => b"Vi",
            b'W' => b"DuBLYu",
            b'X' => b"KS",
            b'Y' => b"Wa",
            _ => b"Zi",
        };
        for &p in name {
            push(&mut out, p);
        }
    }
    out
}

fn push(out: &mut Vec<Ph>, p: Ph) {
    // Collapse a doubled phoneme (from "ll", or digraph + same letter).
    if out.last() == Some(&p) {
        return;
    }
    out.push(p);
}

/// Encode a span of written words as one phoneme sequence.
pub fn encode_span<'a>(words: impl IntoIterator<Item = &'a str>) -> Vec<Ph> {
    let mut out = Vec::new();
    for w in words {
        out.extend(encode_word(w));
    }
    out
}

/// Cost of substituting `a` for `b`.
fn sub_cost(a: Ph, b: Ph) -> f32 {
    if a == b {
        return 0.0;
    }
    let (va, vb) = (is_vowel(a), is_vowel(b));
    if va && vb {
        // i ↔ e is the closest pair (the "Livio" / "Levio" spellings).
        return if matches!((a, b), (b'i', b'e') | (b'e', b'i')) {
            0.2
        } else {
            0.3
        };
    }
    let glide = |v: Ph, g: Ph| matches!((v, g), (b'i', b'Y') | (b'u', b'W'));
    if glide(a, b) || glide(b, a) {
        return 0.2;
    }
    if va || vb {
        return 1.0;
    }
    let pair = |x: Ph, y: Ph| (a == x && b == y) || (a == y && b == x);
    if pair(b'B', b'V')
        || pair(b'P', b'B')
        || pair(b'T', b'D')
        || pair(b'K', b'G')
        || pair(b'F', b'V')
        || pair(b'S', b'Z')
        || pair(b'C', b'J')
        || pair(b'X', b'C')
        || pair(b'X', b'S')
    {
        return 0.4;
    }
    if pair(b'M', b'N') || pair(b'Q', b'T') || pair(b'Q', b'F') || pair(b'Q', b'D') {
        return 0.5;
    }
    if pair(b'L', b'R') || pair(b'L', b'N') {
        return 0.6;
    }
    1.0
}

/// Cost of inserting or deleting `p`.
fn indel_cost(p: Ph) -> f32 {
    if is_vowel(p) || matches!(p, b'H' | b'Y' | b'W') {
        0.4
    } else {
        1.0
    }
}

/// Weighted Levenshtein distance between two phoneme sequences.
/// Two rows on the stack; sequences longer than 63 never match anyway.
pub fn distance(a: &[Ph], b: &[Ph]) -> f32 {
    distance_within(a, b, f32::MAX)
}

/// [`distance`], giving up (returning a value above `limit`) as soon as
/// every cell of a row exceeds `limit` — distances only grow row to row,
/// so most non-matching pairs stop after two or three rows.
pub fn distance_within(a: &[Ph], b: &[Ph], limit: f32) -> f32 {
    const CAP: usize = 64;
    let m = b.len();
    if m >= CAP || a.len() >= CAP {
        return f32::MAX;
    }
    let mut prev = [0.0f32; CAP];
    let mut cur = [0.0f32; CAP];
    for j in 0..m {
        prev[j + 1] = prev[j] + indel_cost(b[j]);
    }
    for &ai in a {
        cur[0] = prev[0] + indel_cost(ai);
        for j in 0..m {
            let del = prev[j + 1] + indel_cost(ai);
            let ins = cur[j] + indel_cost(b[j]);
            let sub = prev[j] + sub_cost(ai, b[j]);
            cur[j + 1] = del.min(ins).min(sub);
        }
        if cur[..=m].iter().all(|&v| v > limit) {
            return f32::MAX;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

/// Distance normalised by the longer sequence: 0 = same sound,
/// ~1 = unrelated. Empty input never matches.
pub fn score(a: &[Ph], b: &[Ph]) -> f32 {
    let len = a.len().max(b.len());
    if a.is_empty() || b.is_empty() {
        return 1.0;
    }
    distance(a, b) / len as f32
}

/// [`score`] when only scores ≤ `max_score` matter; anything above
/// comes back as 1.0 without finishing the table.
pub fn score_within(a: &[Ph], b: &[Ph], max_score: f32) -> f32 {
    let len = a.len().max(b.len());
    if a.is_empty() || b.is_empty() {
        return 1.0;
    }
    let d = distance_within(a, b, max_score * len as f32);
    if d == f32::MAX { 1.0 } else { d / len as f32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(s: &str) -> String {
        String::from_utf8(encode_span(s.split_whitespace())).unwrap()
    }

    fn sc(a: &str, b: &str) -> f32 {
        score(
            &encode_span(a.split_whitespace()),
            &encode_span(b.split_whitespace()),
        )
    }

    #[test]
    fn encodes_the_observed_mishearings() {
        assert_eq!(enc("Livio"), "LiVio");
        assert_eq!(enc("leave you"), "LiVYu");
        assert_eq!(enc("Leevio"), "LiVio");
        assert_eq!(enc("Libio"), "LiBio");
        assert_eq!(enc("give you"), "GiVYu");
        assert_eq!(enc("Claude Code"), "KLoDKoD");
        assert_eq!(enc("cloud code"), "KLuDKoD");
        assert_eq!(enc("LDV"), "LDV");
    }

    #[test]
    fn sounds_alike_scores_low_and_different_sounds_high() {
        assert!(
            sc("leave you", "Livio") <= 0.12,
            "{}",
            sc("leave you", "Livio")
        );
        assert!(sc("Libio", "Livio") <= 0.12);
        assert!(sc("Leevio", "Livio") == 0.0);
        assert!(sc("cloud code", "Claude Code") <= 0.06);
        assert!(sc("clock code", "Claude Code") <= 0.15);
        // A different first consonant is a different word.
        assert!(sc("give you", "Livio") > 0.25);
        // "leave" alone is missing two syllables.
        assert!(sc("leave", "Livio") > 0.12);
        assert!(sc("leave video", "Livio") > 0.25);
        assert!(sc("alone", "Livio") > 0.3);
        // "NVO" spelled out vs "Livio".
        let nvo = encode_letters("NVO");
        assert!(
            score(&nvo, &encode_word("Livio")) <= 0.21,
            "{}",
            score(&nvo, &encode_word("Livio"))
        );
        // "sadly view" carries the cue word: ≈ "said Livio".
        assert!(
            sc("sadly view", "said Livio") <= 0.08,
            "{}",
            sc("sadly view", "said Livio")
        );
    }
}
