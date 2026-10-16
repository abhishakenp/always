//! Retrain-when-away loop for the learned end-of-turn model.
//!
//! One background thread (`eot-retrain`, Darwin background QoS — the
//! per-thread equivalent of `nice`; Darwin has no per-thread nice value).
//! Every `CHECK_EVERY` it trains only when ALL hold:
//! - `learn_from_voice` is on;
//! - he is away: no voice for `AWAY_AFTER`;
//! - thermal pressure is nominal and the 1-min load per core is ≤ 0.9;
//! - the ring gained ≥ `MIN_NEW_ROWS` labelled rows (or a relabel) since
//!   the last attempt (the first attempt after start always runs).
//!
//! Training is single-threaded, mini-batch (`TrainCfg::batch`), and
//! paced: after every ~10 ms of work it sleeps 4× as long (≤ 20 % of one
//! core), and it aborts on any thermal rise, load spike, or his voice.
//!
//! Promotion (the replay-eval win): the candidate is selected by grouped
//! cross-validation with ZERO held-out splits (`eot::select_and_train`)
//! and must cut more ends early than the current model does on the same
//! rows. A current model that split any thought in rows recorded AFTER it
//! was trained (true held-out data, incl. relabelled ends) is rolled back
//! to `last_good` — or to no early cut — before anything else.

use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use crate::always::eot::{self, EotModel, EvalReport, PauseRow, Selected, TrainCfg};
use crate::always::learn;

const CHECK_EVERY: Duration = Duration::from_secs(5 * 60);
const AWAY_AFTER: Duration = Duration::from_secs(10 * 60);
const MIN_NEW_ROWS: u64 = 20;
/// Max share of one core while training.
const DUTY: f64 = 0.2;
const WORK_SLICE: Duration = Duration::from_millis(10);

/// Darwin thermal pressure (0 nominal, 1 moderate, 2 heavy, 3 trapping,
/// 4 sleeping) from `com.apple.system.thermalpressurelevel` — the same
/// signal as `NSProcessInfo.thermalState`, via libSystem's notify API.
pub fn thermal_level() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn notify_register_check(name: *const libc::c_char, out_token: *mut libc::c_int) -> u32;
            fn notify_get_state(token: libc::c_int, state64: *mut u64) -> u32;
        }
        static TOKEN: std::sync::OnceLock<Option<libc::c_int>> = std::sync::OnceLock::new();
        let token = (*TOKEN.get_or_init(|| {
            let mut t: libc::c_int = 0;
            let name = c"com.apple.system.thermalpressurelevel";
            (unsafe { notify_register_check(name.as_ptr(), &mut t) } == 0).then_some(t)
        }))?;
        let mut state = 0u64;
        (unsafe { notify_get_state(token, &mut state) } == 0).then_some(state)
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

pub fn load_per_core() -> f64 {
    let mut l = [0f64; 3];
    let n = unsafe { libc::getloadavg(l.as_mut_ptr(), 3) };
    let cores = std::thread::available_parallelism().map_or(1, |c| c.get()) as f64;
    if n < 1 { 0.0 } else { l[0] / cores }
}

/// CPU time of the calling thread.
pub fn thread_cpu() -> Duration {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
}

/// Duty-cycle + abort conditions between mini-batches.
pub struct Pacer {
    pub duty: f64,
    slice_cpu: Duration,
    thermal0: Option<u64>,
    calls: u64,
    pub slept: Duration,
    pub thermal_max: Option<u64>,
    pub aborted: Option<&'static str>,
    /// Extra abort condition (he is back, …); checked every 16 batches.
    pub stop: Box<dyn FnMut() -> Option<&'static str>>,
}

impl Pacer {
    pub fn new(duty: f64, stop: Box<dyn FnMut() -> Option<&'static str>>) -> Self {
        let t = thermal_level();
        Self {
            duty,
            slice_cpu: thread_cpu(),
            thermal0: t,
            calls: 0,
            slept: Duration::ZERO,
            thermal_max: t,
            aborted: None,
            stop,
        }
    }

    /// `false` = abort.
    pub fn tick(&mut self) -> bool {
        if self.aborted.is_some() {
            return false;
        }
        self.calls += 1;
        if !self.calls.is_multiple_of(16) {
            return true;
        }
        let worked = thread_cpu().saturating_sub(self.slice_cpu);
        if worked >= WORK_SLICE {
            let rest = worked.mul_f64((1.0 - self.duty) / self.duty);
            std::thread::sleep(rest);
            self.slept += rest;
            self.slice_cpu = thread_cpu();
            let t = thermal_level();
            self.thermal_max = self.thermal_max.max(t);
            if t > self.thermal0 || t.is_some_and(|t| t > 0) {
                self.aborted = Some("thermal");
                return false;
            }
            if let Some(why) = (self.stop)() {
                self.aborted = Some(why);
                return false;
            }
        }
        true
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    Promoted(u32),
    Kept,
    RolledBack(Option<u32>),
    RolledBackAndPromoted(u32),
    Aborted(&'static str),
}

#[derive(Debug)]
pub struct Outcome {
    pub rows: usize,
    pub candidate: Option<(f32, EvalReport)>,
    pub current_heldout: Option<EvalReport>,
    pub decision: Decision,
    pub wall: Duration,
    pub cpu: Duration,
    pub slept: Duration,
    pub thermal_max: Option<u64>,
}

/// One retrain over the learn dir `d`: rollback check, train, promote.
pub fn retrain_once(d: &Path, cfg: &TrainCfg, pacer: &mut Pacer) -> anyhow::Result<Outcome> {
    let (t0, c0) = (Instant::now(), thread_cpu());
    let rows: Vec<PauseRow> = learn::load_rows_in(d);
    let st = learn::read_state(d);
    let current = st.current.and_then(|v| learn::read_model(d, v));
    let last_good = st.last_good.and_then(|v| learn::read_model(d, v));

    // 1. Is the current model splitting his thoughts on data it never saw?
    let mut decision = Decision::Kept;
    let mut current_heldout = None;
    let mut live: Option<EotModel> = current.clone();
    if let Some(cur) = &current {
        let rep = eot::evaluate(
            Some(cur),
            eot::labelled(&rows).filter(|r| r.seq > cur.trained_through_seq),
        );
        if rep.model_splits > 0 {
            let lg_ok = last_good.as_ref().is_some_and(|lg| {
                eot::evaluate(
                    Some(lg),
                    eot::labelled(&rows).filter(|r| r.seq > cur.trained_through_seq),
                )
                .model_splits
                    == 0
            });
            let back = learn::rollback_in(d, lg_ok)?;
            live = back.and_then(|v| learn::read_model(d, v));
            decision = Decision::RolledBack(back);
        }
        current_heldout = Some(rep);
    }

    // 2. Candidate from all labelled rows.
    let sel: Option<Selected> = eot::select_and_train(&rows, cfg, &mut || pacer.tick());
    if let Some(why) = pacer.aborted {
        return Ok(Outcome {
            rows: rows.len(),
            candidate: None,
            current_heldout,
            decision: match decision {
                Decision::RolledBack(_) => decision,
                _ => Decision::Aborted(why),
            },
            wall: t0.elapsed(),
            cpu: thread_cpu().saturating_sub(c0),
            slept: pacer.slept,
            thermal_max: pacer.thermal_max,
        });
    }
    let candidate = sel.as_ref().map(|s| (s.margin, s.cv.clone()));
    if let Some(s) = sel {
        let live_early = live
            .as_ref()
            .map_or(0, |m| eot::evaluate(Some(m), eot::labelled(&rows)).early_ends);
        if s.cv.model_splits == 0 && s.cv.early_ends > live_early {
            let mut m = s.model;
            m.note = format!(
                "rows {} margin {:.2} cv splits {}/{} early ends {}/{} cv end-cut p50 {:?} ms",
                rows.len(),
                s.margin,
                s.cv.model_splits,
                s.cv.continuations,
                s.cv.early_ends,
                s.cv.ends,
                s.cv.pct_ms(0.5)
            );
            let v = learn::promote_in(d, m)?;
            decision = match decision {
                Decision::RolledBack(_) => Decision::RolledBackAndPromoted(v),
                _ => Decision::Promoted(v),
            };
        }
    }
    Ok(Outcome {
        rows: rows.len(),
        candidate,
        current_heldout,
        decision,
        wall: t0.elapsed(),
        cpu: thread_cpu().saturating_sub(c0),
        slept: pacer.slept,
        thermal_max: pacer.thermal_max,
    })
}

fn quiet_enough() -> Option<&'static str> {
    if !learn::enabled() {
        return Some("disabled");
    }
    if crate::always::pause::since_last_voice() < AWAY_AFTER {
        return Some("not_away");
    }
    if thermal_level().is_some_and(|t| t > 0) {
        return Some("thermal");
    }
    if load_per_core() > 0.9 {
        return Some("load");
    }
    None
}

/// Start the retrain-when-away loop (daemon start).
pub fn spawn() {
    let _ = std::thread::Builder::new()
        .name("eot-retrain".into())
        .spawn(|| {
            learn::background_qos();
            let check = std::env::var("ALWAYS_EOT_RETRAIN_CHECK_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .map_or(CHECK_EVERY, Duration::from_secs);
            let mut first = true;
            loop {
                std::thread::sleep(check);
                if quiet_enough().is_some() {
                    continue;
                }
                let new = learn::NEW_ROWS.load(Ordering::Relaxed);
                if !first && new < MIN_NEW_ROWS {
                    continue;
                }
                first = false;
                learn::NEW_ROWS.store(0, Ordering::Relaxed);
                let mut pacer = Pacer::new(
                    DUTY,
                    Box::new(|| {
                        (crate::always::pause::since_last_voice() < Duration::from_secs(5))
                            .then_some("voice")
                            .or_else(|| (load_per_core() > 0.9).then_some("load"))
                    }),
                );
                match retrain_once(&learn::dir(), &TrainCfg::default(), &mut pacer) {
                    Ok(o) => {
                        tracing::info!(
                            rows = o.rows,
                            decision = ?o.decision,
                            cv = ?o.candidate.as_ref().map(|(m, r)| (m, r.model_splits, r.continuations, r.early_ends, r.ends)),
                            current_heldout_splits = o.current_heldout.as_ref().map(|r| r.model_splits),
                            wall_ms = o.wall.as_millis() as u64,
                            cpu_ms = o.cpu.as_millis() as u64,
                            slept_ms = o.slept.as_millis() as u64,
                            thermal_max = o.thermal_max,
                            "eot_retrain"
                        );
                        if !matches!(o.decision, Decision::Kept | Decision::Aborted(_)) {
                            learn::reload_model();
                        }
                    }
                    Err(e) => tracing::warn!(error = %e, "eot_retrain_failed"),
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thermal_and_load_read() {
        let t = thermal_level();
        assert!(t.is_some_and(|t| t <= 4), "{t:?}");
        let l = load_per_core();
        assert!((0.0..64.0).contains(&l));
        let a = thread_cpu();
        let mut x = 0u64;
        for i in 0..2_000_000u64 {
            x = x.wrapping_add(i * i);
        }
        std::hint::black_box(x);
        assert!(thread_cpu() > a);
    }

    #[test]
    fn pacer_holds_duty_cycle() {
        let mut p = Pacer::new(0.2, Box::new(|| None));
        p.thermal0 = Some(0); // the duty test must not depend on the room
        let (t0, c0) = (Instant::now(), thread_cpu());
        let mut x = 0u64;
        let mut i = 0u64;
        while t0.elapsed() < Duration::from_millis(400) {
            for _ in 0..20_000 {
                i += 1;
                x = x.wrapping_add(i.wrapping_mul(i));
            }
            if !p.tick() {
                break;
            }
        }
        std::hint::black_box(x);
        let duty = thread_cpu().saturating_sub(c0).as_secs_f64() / t0.elapsed().as_secs_f64();
        if p.aborted != Some("thermal") {
            assert!(duty < 0.35, "duty {duty:.2}");
        }
    }

    /// A live model that split a thought on rows it never saw is rolled
    /// back — past a last_good that splits too, to no early cut at all.
    #[test]
    fn splitting_model_is_rolled_back() {
        use crate::always::eot::{EotProbe, NF};
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/test-tmp")
            .join(format!("learn-rollback-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let always_cut = EotModel {
            probes: vec![EotProbe {
                probe_frames: 8,
                mean: vec![0.0; NF],
                std: vec![1.0; NF],
                w: vec![0.0; NF],
                b: 10.0,
                threshold: 0.5,
            }],
            trained_through_seq: 100,
            ..EotModel::default()
        };
        learn::promote_in(&d, always_cut.clone()).unwrap();
        learn::promote_in(&d, always_cut).unwrap();
        let mut text = String::new();
        for i in 0..3u64 {
            let r = PauseRow {
                file: format!("u{i}"),
                raw_len: 14,
                resumed: true,
                probes: vec![(8, [0.0; NF])],
                seq: 200 + i,
                ..PauseRow::default()
            };
            text.push_str(&serde_json::to_string(&r).unwrap());
            text.push('\n');
        }
        std::fs::write(d.join("pauses.jsonl"), text).unwrap();
        let mut pacer = Pacer::new(0.5, Box::new(|| None));
        pacer.thermal0 = Some(4);
        let o = retrain_once(&d, &TrainCfg::default(), &mut pacer).unwrap();
        assert_eq!(o.decision, Decision::RolledBack(None), "{o:?}");
        assert_eq!(o.current_heldout.unwrap().model_splits, 3);
        assert_eq!(learn::read_state(&d).current, None);
        let _ = std::fs::remove_dir_all(&d);
    }
}
