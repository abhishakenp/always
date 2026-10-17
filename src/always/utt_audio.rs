//! The audio numbers behind one utterance's verdict, handed to a consumer (Iris) with the
//! held final: the speaker-gate score that verified it, the input level before the daemon's
//! gain, the recogniser's confidence. Iris's addressee model reads them as evidence next to
//! the words (`iris-attention/src/addressee/voice.ts`). Missing numbers stay missing: the
//! consumer then decides on the words alone, as before.
use parking_lot::Mutex;

#[derive(Default, Clone, Copy)]
struct Snap {
    gate: Option<f32>,
    peak_dbfs: Option<f32>,
    gain_db: Option<f32>,
    secs: Option<f64>,
}

static SNAP: Mutex<Snap> = parking_lot::const_mutex(Snap {
    gate: None,
    peak_dbfs: None,
    gain_db: None,
    secs: None,
});

/// A new utterance began: forget the last one's numbers.
pub fn reset() {
    *SNAP.lock() = Snap::default();
}

/// The gate score that verified the speaker (window or whole utterance).
pub fn set_gate(score: f32) {
    SNAP.lock().gate = Some(score);
}

/// The preview recogniser's per-utterance level (what `preview_asr_utterance_end` logs).
pub fn set_level(peak_dbfs: f32, gain_db: f32, secs: f64) {
    let mut s = SNAP.lock();
    s.peak_dbfs = Some(peak_dbfs);
    s.gain_db = Some(gain_db);
    s.secs = Some(secs);
}

/// `gate=0.556;peak=-9.1;gain=4.2;conf=0.367;secs=1.71` — only the numbers that exist; empty
/// when there is no gate score (nothing to weigh).
pub fn wire(conf: Option<f32>) -> String {
    let s = *SNAP.lock();
    let Some(gate) = s.gate else { return String::new() };
    let mut out = format!("gate={gate:.3}");
    if let (Some(p), Some(g)) = (s.peak_dbfs, s.gain_db) {
        out.push_str(&format!(";peak={p:.1};gain={g:.1}"));
    }
    if let Some(c) = conf.filter(|c| *c >= 0.0) {
        out.push_str(&format!(";conf={c:.3}"));
    }
    if let Some(d) = s.secs {
        out.push_str(&format!(";secs={d:.2}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_carries_what_exists_and_nothing_without_a_gate_score() {
        reset();
        assert_eq!(wire(Some(0.9)), "");
        set_gate(0.556);
        assert_eq!(wire(None), "gate=0.556");
        set_level(-9.1, 4.2, 1.71);
        assert_eq!(wire(Some(0.367)), "gate=0.556;peak=-9.1;gain=4.2;conf=0.367;secs=1.71");
        reset();
        assert_eq!(wire(None), "");
    }
}
