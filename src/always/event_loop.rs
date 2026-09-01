use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use parking_lot::RwLock;
use tokio::runtime::{Handle, Runtime};

use crate::always::log::{Event, Logger};
use crate::always::speech_action::{
    SpeechAction, classify_transcription, merge_dictation_with, normalize_for_paste_dedupe,
    paste_dedupe_window,
};
use crate::always::{
    AlwaysConfig, audio, auto_enter_countdown, clipboard_watcher, consume_merge, daemon, event,
    filter, idle_watcher, keyboard, mic_watcher, paste, pause, per_app, transcript_stream,
    uds_server, vad,
};
use crate::managers::model_registry::ModelRegistry;
use crate::stt::Transcriber;
use crate::stt_dispatch::{PendingTranscriber, build_transcriber};

/// Active [`Transcriber`] shared between the main loop, the UDS server
/// (which swaps it when the user picks a different model in Settings),
/// and any future hot-reload sources. `RwLock<Arc<dyn Transcriber>>`
/// gives us cheap cloned reads (one Arc bump per utterance) and a
/// trivial swap on user model change — no need to restart the daemon.
pub type ActiveTranscriber = Arc<RwLock<Arc<dyn Transcriber>>>;

/// Active [`AlwaysConfig`] shared between the main loop and the UDS
/// server so Settings changes apply without a daemon restart.
pub type ActiveConfig = Arc<RwLock<AlwaysConfig>>;

/// Whether the daemon should re-fetch the speaker-gate model at startup.
/// True only when the gate is enabled AND a voiceprint is enrolled (so
/// the gate WILL be requested and, without the model, fail closed) AND
/// the model is currently absent. When the gate is off or no voiceprint
/// exists, the gate is never requested, so a missing model is harmless
/// and we must NOT spend 26 MB of bandwidth chasing it.
fn should_refetch_speaker_model(gate_enabled: bool, enrolled: bool, model_present: bool) -> bool {
    gate_enabled && enrolled && !model_present
}

pub fn run(cfg: &AlwaysConfig) -> Result<()> {
    let _pid = daemon::PidGuard::install()?;
    let mut log = Logger::open(&cfg.log_path)?;
    log.write(Event::Start { cfg });
    print_banner(cfg);

    let rt = Runtime::new()?;

    // Install SIGTERM/SIGINT handlers BEFORE any subsystem starts so a
    // signal during init still triggers cleanup. The handler removes
    // the pid + socket files and calls `std::process::exit(0)`.
    daemon::install_signal_handlers(rt.handle());

    // Initialize the auto-enter state from config
    pause::init_auto_enter(cfg.auto_enter);
    crate::always::status_sound::set_setting(cfg.audible_status_sound);

    // Self-heal the speaker-gate model after a cache wipe. macOS can
    // purge ~/Library/Caches between runs; Silero re-materialises from
    // embedded bytes every boot, but the 26 MB WeSpeaker voiceprint
    // model is download-only (fetched at enrollment) and is NOT restored
    // on startup. Missing, the gate loads as "unavailable" and — because
    // it fails CLOSED — drops EVERY utterance as "not enrolled"
    // (score -1.0), silently bricking capture until the user re-enrolls.
    // If the gate is on and a voiceprint is enrolled but the model is
    // gone, re-fetch it in the background: best-effort, never blocking
    // capture startup, and idempotent (ensure_model is a no-op when the
    // file is already present with the right checksum).
    if should_refetch_speaker_model(
        cfg.speaker_gate_enabled,
        crate::always::voiceprint::is_enrolled(),
        crate::always::speaker_embed::model_present(),
    ) {
        std::thread::spawn(|| {
            tracing::warn!("speaker_model_missing_at_startup_refetching");
            match crate::always::speaker_embed::ensure_model() {
                Ok(_) => tracing::info!("speaker_model_restored"),
                Err(e) => tracing::warn!(error = %e, "speaker_model_refetch_failed"),
            }
        });
    }

    // Create shared config early so UDS server can start immediately
    let active_cfg: ActiveConfig = Arc::new(RwLock::new(cfg.clone()));

    // Start UDS server IMMEDIATELY with a placeholder transcriber so the GUI
    // can connect while the real backend loads in the background.
    let (ready_signal, registry_placeholder, active_placeholder) = {
        let cfg_for_uds = Arc::clone(&active_cfg);
        let registry_placeholder = ModelRegistry::new().context("init model registry")?;
        // Pending placeholder blocks transcribe() until the real backend swaps in,
        // so the first utterance after launch waits for init instead of being
        // dropped with a "still initializing" error.
        let (pending, ready_signal) = PendingTranscriber::new();
        let active_placeholder: ActiveTranscriber = Arc::new(RwLock::new(pending));

        #[cfg(unix)]
        {
            let registry_for_uds = registry_placeholder.clone();
            let active_for_uds = Arc::clone(&active_placeholder);
            let _uds_handle = rt.spawn(async move {
                if let Err(e) =
                    uds_server::start_server(registry_for_uds, active_for_uds, cfg_for_uds).await
                {
                    tracing::error!(error = %e, "UDS server error");
                }
            });
        }
        #[cfg(not(unix))]
        {
            tracing::debug!("UDS server not available on this platform");
        }

        (ready_signal, registry_placeholder, active_placeholder)
    };

    // Send initial state events immediately - UDS server will broadcast them
    // to clients as they connect via the initial state in handle_client
    event::global_broadcaster().listening_started();
    if cfg.auto_enter {
        event::global_broadcaster().auto_enter_enabled();
    }

    // Restore persisted focus/pause state now that the broadcaster and UDS
    // server are up (so any events it emits reach connected clients) but
    // before audio gating begins. The function is a no-op when there is
    // nothing to restore; owned by `focus_state`, just invoked here.
    crate::always::focus_state::restore_on_startup();

    // Keyboard hooks are not needed for UDS bind — start after the socket is live.
    keyboard::start_keyboard_listener()?;

    // Load the real transcriber off the hot path (local models can take seconds).
    let cfg_for_build = cfg.clone();
    let registry_for_build = registry_placeholder.clone();
    let active_for_build = Arc::clone(&active_placeholder);
    std::thread::Builder::new()
        .name("transcriber-init".into())
        .spawn(move || {
            match build_transcriber(&cfg_for_build, &registry_for_build) {
                Ok(transcriber) => {
                    *active_for_build.write() = Arc::clone(&transcriber);
                    // Wake any utterance threads that are blocked in the
                    // PendingTranscriber waiting for init to finish.
                    ready_signal.set(transcriber);
                    tracing::info!("transcriber_ready");
                }
                Err(e) => tracing::error!(error = %e, "transcriber_init_failed"),
            }
        })
        .expect("spawn transcriber-init thread");

    // Now do the expensive operations after UDS is accepting connections
    let active = active_placeholder;

    // Prewarm the pooled HTTPS connection to Groq so the first utterance's
    // STT/grammar calls skip the DNS+TCP+TLS handshake (~100-300ms). Fire
    // and forget — a failure here just means the first real call pays the
    // handshake like before.
    let prewarm_key = cfg.groq_stt_api_key.clone();
    let _prewarm_handle = rt.spawn(async move {
        let started = Instant::now();
        let mut req =
            crate::http_client::async_client().get("https://api.groq.com/openai/v1/models");
        if let Some(key) = prewarm_key {
            req = req.header("Authorization", format!("Bearer {key}"));
        }
        match req.send().await {
            Ok(resp) => tracing::info!(
                status = %resp.status(),
                groq_prewarm_ms = started.elapsed().as_millis() as u64,
                "groq_prewarm"
            ),
            Err(e) => tracing::debug!(error = %e, "groq_prewarm_failed"),
        }
    });

    // Heartbeat task: emit Heartbeat every 5s so connected GUI clients can
    // detect a dead/stalled daemon via watchdog timeout.
    let _heartbeat_handle = rt.spawn(async {
        let mut interval = tokio::time::interval(Duration::from_secs(5));
        // First tick fires immediately; skip it to avoid double-emit at startup.
        interval.tick().await;
        loop {
            interval.tick().await;
            event::global_broadcaster().send(event::DaemonEvent::Heartbeat);
        }
    });

    // Spawn the passive correction-capture watcher when the user has
    // opted in via the `passive_correction_capture` preference. The
    // module is a no-op when `enabled == false`; we still call it so
    // the daemon's wiring is the single source of truth and a future
    // toggle from the UI can simply restart the daemon. Failure to
    // load preferences is non-fatal — we treat it as "feature off".
    let passive_correction_enabled = crate::db::open()
        .and_then(|conn| crate::db::get_preferences(&conn))
        .ok()
        .and_then(|p| p.passive_correction_capture)
        .unwrap_or(false);
    clipboard_watcher::spawn_if_enabled(rt.handle(), passive_correction_enabled);

    // Idle-pause watchdog. Spawns at most one task; no-op when
    // `idle_pause_secs == 0`. Lives for the daemon lifetime.
    idle_watcher::spawn(rt.handle(), cfg.idle_pause_secs, cfg.idle_pause_action);

    // Mic-conflict watchdog: dedicated task so a call starting while
    // `record_utterance` blocks (up to 30s waiting for voice) is caught
    // within the poll interval instead of after the wait. Replaces the
    // inline check that used to live at the top of this loop.
    mic_watcher::spawn(rt.handle());

    let mut last_process = Instant::now() - Duration::from_secs(10);
    let mut last_dup_check = Instant::now();
    // True once capture has been gated, until the stale audio that piled
    // up in `rec`'s buffer during the gate has been drained.
    let mut was_gated = false;

    loop {
        if last_dup_check.elapsed() >= Duration::from_secs(30) {
            last_dup_check = Instant::now();
            daemon::reconcile_duplicate_processes();
        }

        // "My Voice" enrollment recording, queued by the UDS server.
        // Runs on THIS thread (the mic pipeline's single-thread
        // invariant), and deliberately before the pause checks —
        // recording a voiceprint from Settings must work even while
        // dictation is paused (that's exactly when users set it up).
        if let Some(step) = crate::always::enrollment::take_pending() {
            let cfg_snapshot = active_cfg.read().clone();
            if let Err(e) = crate::always::enrollment::run_enrollment(&cfg_snapshot, step) {
                tracing::warn!(error = %e, step = step.as_str(), "enrollment_run_failed");
            }
            continue;
        }

        // See `pause::should_gate_capture` — in consume mode per-app/idle/
        // audio-output pause are irrelevant (no focused-app paste target),
        // but a real call (mic conflict) or the user's explicit mute (master
        // pause) must still silence capture even while a stream consumer
        // (Iris) is listening.
        if pause::should_gate_capture() {
            // Remember that capture was silenced, so the audio `rec`
            // queues up meanwhile is dropped rather than transcribed
            // when listening resumes (see the drain below).
            was_gated = true;
            // Wake-on-voice while idle-paused: keep the mic hot enough to
            // detect speech without running the full VAD/transcribe loop.
            if pause::is_idle_auto_paused()
                && !pause::is_any_global_pause()
                && vad::poll_speech_energy(&active_cfg.read()).unwrap_or(false)
            {
                pause::set_idle_auto_paused(false);
                pause::mark_voice_seen();
                let (effective, changed) = pause::recompute_effective();
                event::global_broadcaster().idle_auto_resumed();
                if changed && !effective {
                    event::global_broadcaster().resumed();
                }
                tracing::info!(effective, changed, "idle_wake_on_voice");
                continue;
            }

            // Wake-on-voice while paused ONLY for system-audio playback
            // (YouTube, Spotify, etc.). Unlike the idle case above, we do
            // NOT clear `audio_output_paused` here — the Swift-side
            // monitor only re-asserts it on a CHANGE, not periodically,
            // so clearing it would leave the daemon believing playback
            // stopped and it would keep transcribing the media's own
            // audio until it actually does. Instead: record+transcribe
            // exactly one utterance when real speech energy is detected,
            // then fall back to being paused for the next iteration.
            if pause::is_audio_output_paused()
                && !pause::is_master_paused()
                && !pause::is_mic_conflict_paused()
                && vad::poll_speech_energy(&active_cfg.read()).unwrap_or(false)
            {
                tracing::info!("audio_output_wake_on_voice");
                let transcriber_snapshot = active.read().clone();
                if let Err(e) = process_one(
                    &active_cfg,
                    &mut log,
                    &mut last_process,
                    rt.handle(),
                    &transcriber_snapshot,
                ) {
                    tracing::error!(error = %e, "voice_processing_error");
                    log.write(Event::Error {
                        message: &format!("Voice processing error: {:#}", e),
                    });
                }
                continue;
            }

            std::thread::sleep(Duration::from_millis(100));
            continue;
        }

        // Note: Removed auto_enter config reload that was causing race conditions
        // Auto-enter state is now managed consistently through keyboard shortcuts and settings UI

        // Coming back from a gated stretch (mic taken by another app,
        // user muted, idle-paused): `rec` kept capturing the whole time
        // and nobody was reading it, so its buffer now holds audio from
        // exactly the period Always was supposed to ignore. Throw it
        // away before the first read, or that backlog gets transcribed
        // and pasted the moment listening resumes.
        //
        // Only after a gate — during normal back-to-back dictation the
        // audio queued while the previous utterance transcribed is the
        // user continuing to talk, and must be kept.
        if was_gated {
            was_gated = false;
            if let Ok(recorder_arc) = audio::RecChild::get_or_spawn() {
                let mut recorder = recorder_arc.lock();
                if let Some(rec) = recorder.as_mut() {
                    // Logged even at 0.0: a gate that queued nothing is
                    // itself worth seeing (it means `rec` got no samples
                    // while the other app held the device), and silence
                    // here previously made a working drain look like a
                    // drain that never ran.
                    let dropped_secs = rec.drain_pending();
                    tracing::info!(dropped_secs, "stale_audio_dropped_after_pause");
                }
            }
        }

        // Take a cheap snapshot of the active transcriber per utterance —
        // if the user swaps models mid-session, the next loop iteration
        // picks up the new backend automatically.
        let transcriber_snapshot = active.read().clone();
        if let Err(e) = process_one(
            &active_cfg,
            &mut log,
            &mut last_process,
            rt.handle(),
            &transcriber_snapshot,
        ) {
            tracing::error!(error = %e, "voice_processing_error");
            log.write(Event::Error {
                message: &format!("Voice processing error: {:#}", e),
            });
            // Don't exit - continue running
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}

/// Map a record/transcribe error to a (kind, message) pair for the
/// `TranscriptionFailed` event. `kind` is a stable machine tag the GUI can
/// branch on; `message` is short human-facing text. Best-effort substring
/// matching on the anyhow error chain since the STT layer surfaces opaque strings.
fn classify_transcription_error(err: &anyhow::Error) -> (&'static str, String) {
    let s = format!("{err:#}").to_lowercase();
    if s.contains("401")
        || s.contains("unauthorized")
        || s.contains("invalid api key")
        || s.contains("invalid_api_key")
        || s.contains("api key")
    {
        ("auth", "Invalid or missing Groq API key".to_string())
    } else if s.contains("429")
        || s.contains("rate limit")
        || s.contains("quota")
        || s.contains("too many requests")
    {
        ("quota", "Groq rate limit or quota exceeded".to_string())
    } else if s.contains("timed out")
        || s.contains("timeout")
        || s.contains("dns")
        || s.contains("connect")
        || s.contains("network")
        || s.contains("sending request")
    {
        ("network", "Network error reaching Groq".to_string())
    } else {
        ("error", "Transcription failed".to_string())
    }
}

fn process_one(
    active_cfg: &ActiveConfig,
    log: &mut Logger,
    last_process: &mut Instant,
    rt: &Handle,
    transcriber: &Arc<dyn Transcriber>,
) -> Result<()> {
    let cfg = active_cfg.read();
    // Snapshot the window this utterance is being dictated into, before a
    // single frame is captured. `handle_speech` compares it against the
    // live focused app so a master-pause paste can only ever land back in
    // the app the words came from. See `pause::dictation_origin_app`.
    pause::set_dictation_origin_app(crate::always::focus_state::load());
    let record_result = match vad::record_utterance(&cfg, log, transcriber, rt) {
        Ok(r) => r,
        Err(e) => {
            let (kind, message) = classify_transcription_error(&e);
            event::global_broadcaster().transcription_failed(kind, message);
            emit_utterance_terminal();
            return Err(e).context("failed to record/transcribe utterance");
        }
    };
    let outcome = match record_result {
        vad::RecordResult::Speech {
            text,
            energy,
            transcription,
            timing,
        } => handle_speech(
            &cfg,
            log,
            &text,
            energy,
            &transcription,
            &timing,
            last_process,
            rt,
        ),
        vad::RecordResult::Silence => {
            // Don't log silence events - they're too frequent and not useful
            Ok(())
        }
        vad::RecordResult::Timeout => {
            log.write(Event::Timeout);
            Ok(())
        }
        vad::RecordResult::DroppedLowEnergy { energy } => {
            log.write(Event::DroppedLowEnergy { energy });
            event::global_broadcaster().low_microphone_volume_maybe(energy);
            Ok(())
        }
        vad::RecordResult::DroppedNoise { raw } => {
            tracing::debug!(raw, "dropped_noise");
            log.write(Event::DroppedNoise { raw: &raw });
            Ok(())
        }
        vad::RecordResult::DroppedSpeaker { score } => {
            log.write(Event::DroppedSpeaker {
                score: score as f64,
            });
            Ok(())
        }
        vad::RecordResult::PreemptedByMicConflict { text } => {
            // Deliberately NOT routed through `handle_speech`: that is
            // the paste path, and the app that took the mic is pasting
            // its own transcript of this same speech. Recording the text
            // here is the whole point of the variant — the words are
            // kept, the keystrokes are not sent.
            log.write(Event::PreemptedByMicConflict { text: &text });
            Ok(())
        }
    };
    emit_utterance_terminal();
    outcome
}

/// Terminal overlay events for one utterance cycle. Every exit of
/// `process_one` — paste, drop, silence, error — must leave the GUI with
/// no lingering "Listening"/"Transcribing" state, otherwise a path that
/// announced voice but never pasted re-shows a stale overlay (the
/// "listening comes back after transcribing" bug). The broadcaster only
/// puts these on the wire when the matching start event is still open,
/// so calling this unconditionally is free on idle cycles.
fn emit_utterance_terminal() {
    event::global_broadcaster().voice_activity_ended();
    event::global_broadcaster().transcribing_stopped();
}

#[allow(clippy::too_many_arguments)]
fn handle_speech(
    cfg: &AlwaysConfig,
    log: &mut Logger,
    text: &str,
    energy: f64,
    transcription: &crate::stt::TranscriptionResult,
    timing: &vad::UtteranceTiming,
    last_process: &mut Instant,
    rt: &Handle,
) -> Result<()> {
    let now = Instant::now();
    let action = classify_transcription(cfg, text, transcription, now, *last_process);

    // Cooldown is observable in the decision but does not advance
    // last_process — let the next utterance re-check.
    if !matches!(action, SpeechAction::InCooldown) {
        *last_process = now;
        log.write(Event::Transcribed { text, energy });
    }

    match action {
        SpeechAction::InCooldown => Ok(()),
        SpeechAction::Rejected { reason } => {
            // Privacy: gate raw transcript text behind `should_log_transcripts`
            // — without the gate, every release-build daemon writes user
            // speech to its info-level logs, which then propagate to any
            // central log collector. The structured `Event::Filtered`
            // record below is still written via the audited log path,
            // which honors the same gate.
            if crate::always::telemetry::should_log_transcripts() {
                tracing::info!(text, filter_reason = %reason, "filtered");
            } else {
                tracing::info!(
                    chars = text.chars().count(),
                    filter_reason = %reason,
                    "filtered"
                );
            }
            // Re-derive the structured filter result for logging fidelity.
            let filter_result = filter::should_accept_with_reason(text, cfg);
            log.write(Event::Filtered {
                text,
                energy,
                reason: filter_result,
            });
            pause::set_last_filtered(text);
            // Real speech a filter rule blocked still lands on the clipboard
            // (no paste, no auto-enter) so a wrong filter verdict costs the
            // user a manual ⌘V instead of the whole utterance. Hallucinated
            // utterances below are excluded — whisper noise artifacts would
            // clobber the clipboard on every false VAD trigger.
            match paste::copy_to_clipboard(text.to_string()) {
                Ok(()) => {
                    tracing::info!(chars = text.chars().count(), "filtered_copied_to_clipboard")
                }
                Err(error) => {
                    tracing::warn!(error = %error, "filtered_clipboard_copy_failed")
                }
            }
            event::global_broadcaster().voice_activity_ended();
            event::global_broadcaster().transcription_filtered(reason);
            Ok(())
        }
        SpeechAction::Hallucinated { reason } => {
            if crate::always::telemetry::should_log_transcripts() {
                tracing::info!(text, reason = %reason, "hallucination filtered");
            } else {
                tracing::info!(
                    chars = text.chars().count(),
                    reason = %reason,
                    "hallucination filtered"
                );
            }
            log.write(Event::Filtered {
                text,
                energy,
                reason: filter::FilterReason::HardPhrase(reason.clone()),
            });
            pause::set_last_filtered(text);
            event::global_broadcaster().voice_activity_ended();
            event::global_broadcaster().transcription_filtered(reason);
            Ok(())
        }
        SpeechAction::Paste { text: transformed } => {
            // Log the raw Whisper transcript for debugging STT quality issues
            // Privacy: gate transcript text behind `should_log_transcripts`
            if crate::always::telemetry::should_log_transcripts() {
                tracing::info!(
                    stage = "whisper_raw",
                    transcript = %text,
                    "speech-to-text output"
                );
            } else {
                tracing::info!(
                    chars = text.chars().count(),
                    stage = "whisper_raw",
                    "speech-to-text output"
                );
            }

            // Resume-merge must use the full blocking pipeline — async
            // grammar patch only replaces the last paste via Cmd+Z, which
            // cannot safely rewrite a merged delta mid-countdown.
            //
            // The merge anchor comes from EITHER the auto-enter dictation
            // buffer (countdown still pending) OR the dictation session
            // (recently pasted text in the same app, no auto-enter
            // required). The session is what fixes choppy long-form
            // dictation for users without auto-enter: natural pauses no
            // longer reset every utterance to a fresh capitalized
            // sentence.
            let merge_previous =
                pause::dictation_buffer_text().or_else(crate::always::dictation::session_text);
            let merge_active = merge_previous.is_some();

            // Short-utterance bypass: tiny inputs like "yes", "ok", "done"
            // get pasted verbatim. Longer utterances: acoustic fix-up is
            // sync (fast); LLM grammar runs async after paste when enabled
            // so the user sees text immediately (~300-800ms sooner).
            let grammar_started = Instant::now();
            // Voice snippets: a trigger phrase ("grill me") anywhere in
            // the utterance is replaced with its configured expansion,
            // surrounding words preserved. Checked before the correction
            // stack — expansions are user-authored text and must never
            // be reworded by the grammar LLM.
            let snippet_expansion = crate::always::snippets::expand(&transformed);
            let GrammarOutcome {
                text: final_text,
                source: grammar_source,
                pending: pending_grammar,
            } = if let Some(expanded) = snippet_expansion {
                let expanded_chars = expanded.chars().count();
                if let Some(utterance) = snippet_utterance_for_log(
                    &transformed,
                    crate::always::telemetry::should_log_transcripts(),
                ) {
                    tracing::info!(
                        stage = "snippet_expansion",
                        utterance = %utterance,
                        expanded_chars,
                        "snippet trigger matched — pasting expansion verbatim"
                    );
                } else {
                    tracing::info!(
                        stage = "snippet_expansion",
                        utterance_chars = transformed.chars().count(),
                        expanded_chars,
                        "snippet trigger matched — pasting expansion verbatim"
                    );
                }
                // Snippet expansions are user-authored text. They are
                // pasted verbatim and never handed to the grammar LLM —
                // no request is built, so no patch can rewrite them.
                GrammarOutcome::bypass(expanded)
            } else if is_short_utterance(&transformed) {
                tracing::info!(
                    stage = "short_utterance_bypass",
                    text = %transformed,
                    "skipping correction stack — too short to benefit"
                );
                GrammarOutcome::bypass(transformed.clone())
            } else if transformed.chars().count() > GRAMMAR_MAX_CHARS {
                // Long chunked dictation: a single blocking LLM pass
                // over the joined text would blow the 8s grammar
                // timeout. Each chunk was already grammar-corrected
                // in the background as it flushed (chunker.rs), so
                // paste the join as-is.
                tracing::info!(
                    stage = "long_transcript_grammar_bypass",
                    chars = transformed.chars().count(),
                    "skipping blocking grammar — per-chunk corrections already applied"
                );
                GrammarOutcome::bypass(transformed.clone())
            } else {
                // The LLM is OFF the critical path (`grammar_wait_ms`
                // defaults to 0). We probe the correction cache — which
                // costs a mutex lock, not a round-trip — and paste
                // whatever we have. On a miss the acoustically corrected
                // text goes out immediately and the LLM keeps running in
                // the background.
                //
                // This replaced a synchronous `block_on` with an 8 s
                // ceiling whose measured cost was p50 1 061 ms / p90
                // 2 464 ms / max 7 783 ms (n=554 utterances that reached
                // the LLM) of dead time before a single character
                // appeared. The single-paste guarantee is unchanged:
                // exactly one paste happens here either way.
                //
                // The request bundles tier-1 acoustic rewrites, deferred
                // fuzzy glossary candidates, and dictation-session context
                // into one LLM message. The speculative warm in `vad.rs`
                // builds the identical request, so an already-finished
                // warm lands as a cache hit and is applied to this same
                // single paste at zero cost.
                let llm_available = cfg.postprocess_available();
                let req = crate::always::correction_request::build(&transformed, llm_available);
                apply_grammar_nonblocking(req, cfg, rt)
            };
            let grammar_ms = grammar_started.elapsed().as_millis() as u64;
            let grammar_cache_hit = grammar_source == GrammarSource::Cache;

            // Privacy: gate transcript text behind `should_log_transcripts`
            if crate::always::telemetry::should_log_transcripts() {
                tracing::info!(
                    stage = "final_result",
                    text = %final_text,
                    energy = energy,
                    "transcript ready for pasting"
                );
            } else {
                tracing::info!(
                    stage = "final_result",
                    chars = final_text.chars().count(),
                    energy = energy,
                    "transcript ready for pasting"
                );
            }
            log.write(Event::Pasting {
                raw: text,
                processed: &final_text,
                energy,
            });
            // Reference point for `pre_paste_ms` below: everything between
            // here and the clipboard write (merge, dedupe, stream append,
            // pause checks) should be near-instant — unexplained
            // multi-second stalls were observed in this window and this
            // makes them measurable.
            let pasting_logged_at = Instant::now();

            // Read the lease ONCE. Both the broadcast below and the routing
            // decision after it depend on it, and a lease that dropped
            // between two reads would either skip the final entirely or
            // emit a fragment final that `consume_merge` then duplicates.
            let consuming = pause::is_consume_mode();

            // Consume mode owns its own commit: `consume_merge` rejoins
            // fragments of one request and emits the single `TranscriptFinal`
            // itself, so broadcasting a per-fragment final here would deliver
            // the pieces AND the whole (the consumer's dedupe is an exact
            // string match and would not collapse them).
            if !consuming {
                event::global_broadcaster().transcript_final(final_text.clone());
            }

            // Route to Iris: the transcript has already been broadcast over UDS
            // (TranscriptChunk previews during speech + the cumulative chunk
            // `consume_merge` publishes per fragment). Then STOP — no
            // clipboard, no paste, no Enter. Nothing is inserted into any app.
            // A persisted stream setting is not proof that a controller is
            // connected. Only a live per-connection consume lease may suppress
            // paste; otherwise a controller crash could silently drop a wake
            // word instead of dictating it normally.
            if consuming {
                // Fold into the request being assembled. This publishes the
                // cumulative text as a `TranscriptChunk` now and commits one
                // `TranscriptFinal` (plus the stream line) once the user has
                // stopped continuing — the same rejoining the paste path
                // below does, which consume mode used to return above.
                consume_merge::accept(
                    rt,
                    &cfg.localization,
                    cfg.transcript_stream_enabled,
                    &final_text,
                );
                pause::dictation_buffer_clear();
                event::global_broadcaster().voice_activity_ended();
                tracing::info!(
                    chars = final_text.chars().count(),
                    "consume_mode_routed_to_stream"
                );
                return Ok(());
            }

            // Resume-merge path: if the auto-enter countdown is still
            // active from the previous utterance, the user paused only
            // briefly and is now continuing. Append the new transcript
            // to the existing buffered text (preserving sentence casing
            // — see `merge_dictation`) instead of pasting it as a fresh,
            // re-capitalized sentence. Cancels the in-flight countdown
            // and reschedules so the timer restarts from the resumed
            // speech, giving the user the full delay to keep going.
            // Merge whenever the dictation buffer still holds the
            // previous paste — not just when the countdown is still
            // ticking. The voice-activity hook in `vad.rs` cancels
            // the countdown as soon as the user resumes speaking,
            // which previously broke the merge gate (countdown_active
            // flipped to false before the new utterance finalised) and
            // produced split-sentence pastes. The buffer is still
            // cleared on Return commit / pause / explicit user cancel,
            // so this can't leak across sessions.
            let (paste_clipboard, buffer_text) = if merge_active {
                let previous = merge_previous.unwrap_or_default();
                let (joined, delta) =
                    merge_dictation_with(&cfg.localization, &previous, &final_text);
                tracing::info!(
                    stage = "dictation_merge",
                    previous = %previous,
                    addition = %final_text,
                    delta = %delta,
                    joined = %joined,
                    "merging resumed utterance into in-flight dictation"
                );
                pause::countdown_request_cancel();
                // Delta carries its own leading space when needed; no
                // trailing space here because we want the merged buffer
                // to end exactly at the last character pasted so a
                // subsequent merge picks up correct sentence-end state.
                (delta, joined)
            } else {
                // Fresh paste — preserve the historical trailing space
                // on the clipboard payload (clients expect it).
                (format!("{} ", final_text), final_text.clone())
            };

            let dup_window = paste_dedupe_window(cfg.cooldown_ms);
            let paste_payload = paste_clipboard.trim();
            if pause::should_suppress_duplicate_paste(paste_payload, dup_window) {
                tracing::info!(
                    text = %paste_payload,
                    dup_window_ms = dup_window.as_millis() as u64,
                    "duplicate_paste_suppressed"
                );
                event::global_broadcaster().voice_activity_ended();
                return Ok(());
            }

            if !pause::try_begin_paste() {
                tracing::info!(
                    text = %paste_payload,
                    "duplicate_paste_suppressed_in_flight"
                );
                event::global_broadcaster().voice_activity_ended();
                return Ok(());
            }

            // External transcript stream (IRIS ears): after dedup so a
            // suppressed double-paste is never double-streamed, but before
            // the paused / cmd-held drops — the user DID speak, only the
            // paste is withheld there.
            if cfg.transcript_stream_enabled {
                transcript_stream::append(&final_text);
            }

            // Focus moved to a paused app (or master/idle pause kicked in)
            // between when we started recording and now. Drop the paste so
            // the transcript doesn't leak into the wrong window once the
            // user gets there. Surface as `transcription_filtered` so the
            // GUI flashes a "not pasted" overlay — same channel as the
            // cmd-held drop below.
            //
            // EXCEPTION: when MASTER pause is the only active source AND
            // focus never left the app this utterance was dictated into,
            // paste anyway. The user hitting their own mute means "stop
            // listening", not "throw away what I already said" — and with
            // focus unchanged there is no other window to leak into, which
            // is the entire risk the drop above exists to prevent. Every
            // other pause source (per-app, idle, mic-conflict,
            // audio-output, no-GUI) keeps the drop unchanged.
            let master_mute_same_app = pause::paused_only_by_master()
                && pause::dictation_origin_app().is_some_and(|origin| {
                    crate::always::focus_state::load().as_deref() == Some(origin.as_str())
                });
            if pause::is_paused() && master_mute_same_app {
                tracing::info!(
                    text = %paste_payload,
                    "paste_kept_master_pause_focus_unchanged"
                );
            }
            if pause::is_paused() && !master_mute_same_app {
                tracing::info!(
                    text = %paste_payload,
                    "paste_dropped_paused_mid_utterance"
                );
                log.write(Event::Error {
                    message: "Skipped paste: app paused mid-utterance",
                });
                // Keep the corrected text recoverable via the force-paste
                // shortcut — this drop is the most common "why did my
                // dictation vanish" path.
                pause::set_last_filtered(final_text.clone());
                event::global_broadcaster()
                    .transcription_filtered("App paused — press ⌃⌥V to paste anyway");
                pause::dictation_buffer_clear();
                pause::end_paste();
                return Ok(());
            }

            // NOTE: no duplicate-daemon check here. It used to fork
            // `ps -eo pid=,args=` on EVERY paste (multi-second stalls were
            // traced to this window); the 30s reconcile loop at startup
            // already owns duplicate-process cleanup.

            // Inserted (dictation) text is LEFT on the clipboard on purpose —
            // the user wants every pasted utterance to land in their clipboard
            // history. So we no longer snapshot/restore the prior clipboard
            // here. (Consume-mode / Iris-routed utterances returned above and
            // never reach this copy, so Iris speech never touches the
            // clipboard — see the `is_consume_mode()` early return.)
            let paste_started = Instant::now();
            // Keep a copy of the payload we're about to write for `note_pasted`
            // below (`paste_clipboard` itself is moved into copy_to_clipboard).
            let written_clipboard = paste_clipboard.clone();

            // NOTE: the paste-in-flight lock is held from `try_begin_paste()`
            // above. Every early-return below MUST release it via
            // `end_paste()` or the daemon goes permanently mute (future
            // utterances drop as "in_flight"). A bare `?` here previously
            // leaked the lock when pbcopy failed.
            let copy_result = paste::copy_to_clipboard(paste_clipboard);
            if let Err(err) = copy_result {
                tracing::warn!(error = %err, "clipboard_copy_failed");
                log.write(Event::Error {
                    message: "Skipped paste: clipboard copy failed",
                });
                pause::set_last_filtered(final_text.clone());
                event::global_broadcaster().transcription_filtered("Clipboard error — not pasted");
                event::global_broadcaster().voice_activity_ended();
                pause::dictation_buffer_clear();
                pause::end_paste();
                return Ok(());
            }

            // Skip paste if Command is held — likely a shortcut in flight.
            // Surface the drop on the status overlay (same channel as
            // hallucination / hard-filter rejections) so the user knows
            // their utterance was heard but intentionally not pasted.
            if keyboard::is_cmd_held() {
                tracing::debug!("skipped_paste_cmd_held");
                log.write(Event::Error {
                    message: "Skipped paste: Command key held",
                });
                pause::set_last_filtered(final_text.clone());
                event::global_broadcaster().transcription_filtered("Held Command key — not pasted");
                pause::dictation_buffer_clear();
                pause::end_paste();
                return Ok(());
            }

            // Always paste WITHOUT enter — the auto-enter Return key
            // is now decoupled, gated behind an optional countdown so
            // the user can intercept (any key cancels). When
            // `auto_enter_delay_ms == 0` the countdown helper presses
            // Return immediately, preserving the legacy behavior.
            if let Err(err) = paste::paste_text(false) {
                pause::end_paste();
                return Err(err);
            }
            let pasted_at = Instant::now();
            // One structured line per pasted utterance so production logs
            // expose where speech-end → paste time goes. `stt_wait_ms` is 0
            // when speculative STT beat the final-silence cut; `grammar_ms`
            // near 0 means the speculative grammar warm landed in cache.
            tracing::info!(
                stage = "latency_breakdown",
                stt_wait_ms = timing
                    .stt_done_at
                    .saturating_duration_since(timing.speech_end_at)
                    .as_millis() as u64,
                pipeline_ms = grammar_started
                    .saturating_duration_since(timing.stt_done_at)
                    .as_millis() as u64,
                grammar_ms,
                pre_paste_ms = paste_started
                    .saturating_duration_since(pasting_logged_at)
                    .as_millis() as u64,
                paste_ms = pasted_at
                    .saturating_duration_since(paste_started)
                    .as_millis() as u64,
                total_ms = pasted_at
                    .saturating_duration_since(timing.speech_end_at)
                    .as_millis() as u64,
                speculation_used = timing.speculation_used,
                grammar_cache_hit,
                // `grammar_cache_hit` alone could never distinguish a cold
                // call from a single-flight join to a still-running warm —
                // both logged `false` — which is why warm effectiveness was
                // unmeasurable. `grammar_source` names the actual path.
                grammar_source = grammar_source.as_str(),
                "utterance latency"
            );
            // Snapshot the focused app at paste time. If the user switches
            // apps (or submits the message) before the grammar patch fires,
            // Cmd+Z would land in the wrong window and create a double paste.
            let pasted_to_app = pause::current_app();

            // Add a small delay after paste to prevent double-paste when
            // the user makes a break mid-utterance. This gives the system
            // time to settle before releasing the paste-in-flight lock.
            // 100ms (was 200): the Cmd+V key event has already been posted
            // and consumed by the target app well within this window; the
            // extra 100ms only delayed readiness for the next utterance.
            std::thread::sleep(Duration::from_millis(100));

            // Release the paste-in-flight lock HERE, unconditionally, before
            // any background grammar work. The retired async path held it
            // inside the spawned task for the whole LLM round-trip, which
            // turned ~1 s of LLM latency into ~1 s of deafness: an utterance
            // finalizing in that window was dropped outright
            // ("duplicate_paste_suppressed_in_flight"), not queued, not even
            // recoverable via force-paste.
            pause::end_paste();
            // The pasted transcript stays on the clipboard so it enters the
            // user's clipboard history. We intentionally do NOT restore the
            // prior clipboard here.

            let auto_enter_effective =
                per_app::effective_auto_enter(pause::is_auto_enter_enabled());
            if auto_enter_effective && should_auto_enter_for_text(&buffer_text) {
                // Hold the merged buffer so the NEXT utterance can
                // append again if it arrives before the countdown
                // fires. Set BEFORE schedule() so even a 0-delay
                // immediate-Return path sees the buffer for clearing.
                pause::dictation_buffer_set(buffer_text.clone());
                let delay = per_app::effective_auto_enter_delay_ms(cfg.auto_enter_delay_ms);
                auto_enter_countdown::schedule(rt, delay);
            } else {
                if auto_enter_effective {
                    tracing::info!(
                        text = %buffer_text,
                        words = word_count(&buffer_text),
                        "auto_enter_suppressed_short_transcript"
                    );
                }
                // No auto-enter, or transcript too short to submit — no
                // merge window. Drop any stale buffer so the next utterance
                // doesn't accidentally try to merge with text the user has
                // long since committed.
                pause::dictation_buffer_clear();
            }

            // Snapshot the freshly-pasted text as the diff baseline for
            // the manual-correction capture pipeline. We store the
            // post-vocabulary, post-postprocess `final_text` (i.e. the
            // exact bytes the user sees in their app) so the ⌃⌥X hotkey
            // and passive clipboard watcher can compare it against the
            // user's selection without having to reconstruct what was
            // pasted. Only runs on the actually-pasted branch — if
            // Cmd was held the early-return above skipped paste and
            // there's nothing for a correction-diff to anchor on.
            // For merge: store the full joined text so correction tools
            // diff against what's actually on screen.
            pause::set_last_pasted(buffer_text.clone());
            // Refresh the dictation session with the authoritative
            // on-screen text. Runs AFTER the auto-enter branch (which may
            // clear buffers) — replace semantics make the ordering safe.
            // The next utterance within the session window joins as a
            // continuation and the grammar LLM sees this as context.
            //
            // Fresh pastes use the CLIPBOARD payload (trailing space
            // included) so the session mirrors what's actually on screen
            // — merging against a space-stripped buffer would add a
            // second space on every continuation. Merge pastes use the
            // joined text, whose spacing already chains correctly.
            if merge_active {
                crate::always::dictation::note_pasted(&buffer_text);
            } else {
                crate::always::dictation::note_pasted(&written_clipboard);
            }

            // In-place grammar patch (opt-in, `grammar_patch_after_paste`).
            //
            // Spawned LAST, after every anchor above is settled, so the
            // patch's own bookkeeping replaces a consistent snapshot rather
            // than racing the paste path that created it. A merge paste is
            // excluded: `replace_via_undo` can only take back the last
            // paste, and on a merge that is the delta, not the joined text
            // the patch would be rewriting.
            if let Some(pending) = pending_grammar {
                if merge_active {
                    tracing::debug!(
                        stage = "grammar_patch",
                        "skipped — merge paste, undo would take back only the delta"
                    );
                } else {
                    spawn_grammar_patch(rt, cfg, pending, pasted_at, pasted_to_app);
                }
            }

            // Explicit voice-activity-ended after a successful paste so the
            // Swift overlay clears cleanly. Without this, the next VAD loop
            // iteration can pick up residual mic energy and fire a new
            // VoiceActivityDetected before the overlay from this utterance
            // has hidden — causing the "listening appears again" flash.
            event::global_broadcaster().voice_activity_ended();
            Ok(())
        }
    }
}

/// Maximum word count for the short-utterance bypass. Inputs at or below
/// this go straight to paste without the acoustic+LLM correction stack.
pub const SHORT_UTTERANCE_MAX_WORDS: usize = 2;

/// Maximum character count (after trim) for the short-utterance bypass.
/// Belt-and-braces companion to `SHORT_UTTERANCE_MAX_WORDS` so a single
/// long word ("acknowledged") still flows through correction, but a
/// two-letter "ok" does not.
pub const SHORT_UTTERANCE_MAX_CHARS: usize = 8;

/// True when the transcript is short enough to skip the correction stack
/// (`≤ SHORT_UTTERANCE_MAX_WORDS` words OR `≤ SHORT_UTTERANCE_MAX_CHARS`
/// trimmed chars). Empty input also returns true — nothing to correct.
pub fn is_short_utterance(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    let words = word_count(trimmed);
    let chars = trimmed.chars().count();
    words <= SHORT_UTTERANCE_MAX_WORDS || chars <= SHORT_UTTERANCE_MAX_CHARS
}

/// Above this size the blocking grammar pass is skipped: the 8s LLM
/// timeout can't absorb a multi-minute transcript, and chunked dictations
/// (the only realistic source of text this long) were already corrected
/// per-chunk while recording (see `chunker.rs`).
pub const GRAMMAR_MAX_CHARS: usize = 4000;

// One word is enough. The user dictates single-word commands ("yes", "run",
// "send") and expects Return to fire — suppressing auto-enter below 3 words
// left one-word utterances stranded without a newline. `>= 1` still excludes
// the empty/whitespace case (word_count 0).
const AUTO_ENTER_MIN_WORDS: usize = 1;

fn should_auto_enter_for_text(text: &str) -> bool {
    word_count(text) >= AUTO_ENTER_MIN_WORDS
}

fn snippet_utterance_for_log(utterance: &str, log_transcripts: bool) -> Option<&str> {
    log_transcripts.then_some(utterance)
}

fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Mirrors Iris's JS-side wake-word matcher (`AlwaysEventClient`, default
/// word "iris"): true when the leading token is exactly the wake word,
/// case-insensitively, followed by nothing / a space / a comma. Used only
/// as a race-guard heuristic (see the call site in `handle_speech`) — the
/// authoritative routing decision remains `pause::is_consume_mode()`.
/// Where the pasted text's grammar came from, for latency attribution.
///
/// `grammar_cache_hit` alone was ambiguous: a cold LLM call and a
/// single-flight join to a still-running speculative warm both reported
/// `false`, so the warm's effectiveness could not be measured from logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarSource {
    /// No LLM involved: snippet, short utterance, oversized transcript,
    /// or grammar correction unavailable for this backend.
    Bypass,
    /// Correction was already in the cache — applied at zero cost to the
    /// same single paste. This is the outcome a working warm produces.
    Cache,
    /// The LLM landed inside a non-zero `grammar_wait_ms` budget.
    Waited,
    /// Budget expired (or was 0). The acoustic text was pasted and the
    /// LLM is still running in the background.
    Deferred,
}

impl GrammarSource {
    pub fn as_str(self) -> &'static str {
        match self {
            GrammarSource::Bypass => "bypass",
            GrammarSource::Cache => "cache",
            GrammarSource::Waited => "waited",
            GrammarSource::Deferred => "deferred",
        }
    }
}

/// A grammar correction that outlived the paste.
pub struct PendingGrammar {
    /// Still-running LLM call. Spawned onto the runtime (not awaited
    /// inline) so that abandoning the wait cannot cancel it: a dropped
    /// future would also leave the single-flight cell uninitialised and
    /// throw the work away instead of banking it in the cache.
    handle: tokio::task::JoinHandle<Option<String>>,
    /// Exactly what we pasted, for the no-op check and the undo guard.
    acoustic: String,
}

/// What the paste path should insert, plus any correction still in
/// flight behind it.
pub struct GrammarOutcome {
    pub text: String,
    pub source: GrammarSource,
    pub pending: Option<PendingGrammar>,
}

impl GrammarOutcome {
    /// Text that never reaches the LLM — pasted exactly as given.
    pub fn bypass(text: String) -> Self {
        Self {
            text,
            source: GrammarSource::Bypass,
            pending: None,
        }
    }
}

/// Grammar cleanup that never blocks the user.
///
/// Ordering, and why:
/// 1. **Cache probe** (a mutex lock, no I/O). A finished speculative warm
///    lands here and is applied to the same single paste for free.
/// 2. **Spawn** the correction onto the runtime, so it survives whatever
///    we do next. The previous implementation awaited the future directly
///    under `block_on(timeout(8s, ..))`; on expiry the future was dropped,
///    which cancelled the in-flight request AND left the single-flight
///    `OnceCell` uninitialised, so the round-trip already paid for was
///    discarded rather than cached.
/// 3. **Wait at most `grammar_wait_ms`** — default 0, i.e. not at all.
///
/// Returns the acoustic text on every path where the LLM has not already
/// answered, so the caller always has something to paste immediately.
fn apply_grammar_nonblocking(
    req: crate::always::correction_request::CorrectionRequest,
    cfg: &AlwaysConfig,
    rt: &Handle,
) -> GrammarOutcome {
    let fallback = req.acoustic_text.clone();
    if !cfg.postprocess_available() {
        tracing::debug!(
            stage = "grammar_correction",
            text = %fallback,
            "grammar correction disabled"
        );
        return GrammarOutcome::bypass(fallback);
    }
    let Some(pp) = cfg.post_processor.clone() else {
        return GrammarOutcome::bypass(fallback);
    };

    // 1. Free hit: the warm already paid for this exact request.
    if let Some(cached) = pp.cached_correction(&req.user_message) {
        if cached != fallback {
            tracing::info!(
                stage = "grammar_correction",
                before = %fallback,
                after = %cached,
                elapsed_ms = 0_u64,
                cache_hit = true,
                "grammar correction applied from cache"
            );
        }
        return GrammarOutcome {
            text: cached,
            source: GrammarSource::Cache,
            pending: None,
        };
    }

    // 2. Detached so the wait budget can never cancel the work.
    let joined_inflight = pp.has_inflight(&req.user_message);
    let started = Instant::now();
    let task_fallback = fallback.clone();
    let handle = rt.spawn(async move {
        match pp.process_request(&req).await {
            Ok((corrected, _cache_hit)) => Some(corrected),
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    fallback_text = %task_fallback,
                    "grammar correction failed, acoustic match result kept"
                );
                None
            }
        }
    });

    // 3. Optional budget. Zero by default — the user never waits.
    let budget = Duration::from_millis(cfg.postprocess_config.grammar_wait_ms);
    if !budget.is_zero() {
        let mut handle = handle;
        let waited = rt.block_on(async { tokio::time::timeout(budget, &mut handle).await });
        match waited {
            Ok(Ok(Some(corrected))) => {
                tracing::info!(
                    stage = "grammar_correction",
                    elapsed_ms = started.elapsed().as_millis() as u64,
                    joined_inflight,
                    changed = corrected != fallback,
                    "grammar correction landed inside wait budget"
                );
                return GrammarOutcome {
                    text: corrected,
                    source: GrammarSource::Waited,
                    pending: None,
                };
            }
            // Task finished with no usable correction (error arm above, or
            // the task itself panicked / was aborted): nothing to patch.
            Ok(_) => {
                return GrammarOutcome {
                    text: fallback,
                    source: GrammarSource::Waited,
                    pending: None,
                };
            }
            Err(_elapsed) => {
                tracing::debug!(
                    stage = "grammar_correction",
                    budget_ms = budget.as_millis() as u64,
                    "wait budget expired — pasting acoustic text, LLM still running"
                );
                return GrammarOutcome {
                    text: fallback.clone(),
                    source: GrammarSource::Deferred,
                    pending: Some(PendingGrammar {
                        handle,
                        acoustic: fallback,
                    }),
                };
            }
        }
    }

    GrammarOutcome {
        text: fallback.clone(),
        source: GrammarSource::Deferred,
        pending: Some(PendingGrammar {
            handle,
            acoustic: fallback,
        }),
    }
}

/// How much of a change is worth taking the user's text back for.
///
/// A pure capitalisation or trailing-punctuation fix is not worth a
/// visible undo-and-repaste flicker in the user's editor; a word-level
/// change is. Measured over 447 real corrections in this user's logs:
/// 9.2% case-only, 34.2% punctuation-or-case-only, 56.6% word-level.
fn patch_is_worth_applying(before: &str, after: &str) -> bool {
    if before == after {
        return false;
    }
    if normalize_for_paste_dedupe(before) == normalize_for_paste_dedupe(after) {
        return false;
    }
    word_sequence(before) != word_sequence(after)
}

/// Lowercased alphanumeric word sequence — the unit `patch_is_worth_applying`
/// compares. Punctuation and casing deliberately fall out.
fn word_sequence(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

/// After paste-first, patch grammar via Cmd+Z + repaste when the LLM returns.
///
/// **Opt-in** (`grammar_patch_after_paste`, default off). The undo step is
/// what retired this path in `34bb5d1`: it doubled the transcript whenever
/// the undo failed to land — the user pressed Return first, the app has
/// non-standard undo (Slack, terminals, web contenteditable), or focus
/// shifted. `dictation.rs` records the resulting rule: "Forward-only by
/// design: text already pasted is never retro-edited."
///
/// Every guard hole in the retired implementation is closed here:
/// - it reuses the ALREADY-RUNNING correction for the same request rather
///   than issuing a second `pp.process(text, None)` call, which built a
///   different cache key and silently dropped the glossary candidates and
///   dictation context from the prompt;
/// - the paste-in-flight lock is released by the caller before this is
///   spawned, so LLM latency is no longer deafness;
/// - `LAST_PASTED` and the dictation session are updated BEFORE the
///   clipboard write, closing the window in which the passive clipboard
///   watcher diffed corrected text against a stale acoustic anchor and
///   laundered a machine edit into a user-approved glossary entry;
/// - the daemon's own synthetic Cmd+Z / Cmd+V are marked so its own
///   keyboard tap does not cancel the auto-enter countdown;
/// - the "message already submitted" guard uses the PER-APP effective
///   auto-enter setting, not the global preference;
/// - a zero auto-enter delay aborts the patch outright: that countdown
///   path presses Return after 50 ms with no cancel check at all, so no
///   patch can win the race.
fn spawn_grammar_patch(
    rt: &Handle,
    cfg: &AlwaysConfig,
    pending: PendingGrammar,
    pasted_at: Instant,
    pasted_to_app: Option<String>,
) {
    if !cfg.postprocess_config.grammar_patch_after_paste {
        // Still drain the handle so the correction reaches the cache — the
        // next identical utterance then gets it for free.
        rt.spawn(async move {
            let _ = pending.handle.await;
        });
        return;
    }

    let auto_enter_effective = per_app::effective_auto_enter(pause::is_auto_enter_enabled());
    let auto_enter_delay_ms = per_app::effective_auto_enter_delay_ms(cfg.auto_enter_delay_ms);
    if auto_enter_effective && auto_enter_delay_ms == 0 {
        tracing::debug!(
            stage = "grammar_patch",
            "skipped — immediate auto-enter, Return fires before any patch can land"
        );
        rt.spawn(async move {
            let _ = pending.handle.await;
        });
        return;
    }

    let PendingGrammar { handle, acoustic } = pending;
    rt.spawn(async move {
        let started = Instant::now();
        let Ok(Some(cleaned)) = handle.await else {
            return;
        };

        if !patch_is_worth_applying(&acoustic, &cleaned) {
            tracing::debug!(
                stage = "grammar_patch",
                elapsed_ms = started.elapsed().as_millis() as u64,
                "no change worth an undo"
            );
            return;
        }

        if pasted_at.elapsed() > paste::GRAMMAR_PATCH_MAX_AGE {
            tracing::debug!(
                stage = "grammar_patch",
                elapsed_since_paste_ms = pasted_at.elapsed().as_millis() as u64,
                "skipped — paste too old to safely undo"
            );
            return;
        }

        if keyboard::is_cmd_held() {
            tracing::debug!("grammar_patch_skipped_cmd_held");
            return;
        }

        // Abort if the focused app changed since the paste. The most common
        // case: user pressed Enter to submit the message before the LLM
        // corrected it. Cmd+Z would land in the wrong window (e.g., a code
        // editor) and the subsequent Cmd+V would create a duplicate paste.
        let current_app = pause::current_app();
        if current_app != pasted_to_app {
            tracing::info!(
                pasted_to = ?pasted_to_app,
                current = ?current_app,
                "grammar_patch_aborted_app_switched"
            );
            return;
        }

        // Someone pasted after us — ours is no longer the last paste, so
        // Cmd+Z would take back THEIR text. The retired implementation had
        // no such check because it held the paste lock across the LLM call
        // instead, at the cost of dropping every utterance in that window.
        if pause::last_pasted_text_within(paste::GRAMMAR_PATCH_MAX_AGE).as_deref()
            != Some(acoustic.as_str())
        {
            tracing::info!("grammar_patch_aborted_superseded_by_newer_paste");
            return;
        }

        // Abort when auto-enter is active and the dictation buffer was
        // cleared — auto_enter_countdown clears it the moment Return fires.
        // If the buffer is gone, the message was already submitted: Cmd+Z
        // would do nothing (empty input) and Cmd+V would ghost-paste the
        // corrected text as a new message — the "double transcript" bug.
        if auto_enter_effective && pause::dictation_buffer_text().is_none() {
            tracing::info!("grammar_patch_aborted_message_submitted");
            return;
        }

        // Take the paste lock so a concurrently finalizing utterance cannot
        // paste into the middle of our undo+repaste. Held for the keystroke
        // sequence only (tens of ms), never across the LLM call.
        if !pause::try_begin_paste() {
            tracing::info!("grammar_patch_aborted_paste_in_flight");
            return;
        }
        struct PasteRelease;
        impl Drop for PasteRelease {
            fn drop(&mut self) {
                pause::end_paste();
                pause::end_synthetic_input();
            }
        }
        let _paste_release = PasteRelease;

        // Update the anchors BEFORE touching the clipboard. The passive
        // clipboard watcher polls every 250 ms and diffs "clipboard now"
        // against `LAST_PASTED`; with the old ordering it could observe the
        // corrected clipboard against the stale acoustic anchor and file the
        // LLM's own edit as a user correction, badging it for glossary
        // promotion.
        let clipboard = format!("{cleaned} ");
        pause::set_last_pasted(cleaned.clone());
        crate::always::dictation::note_pasted(&clipboard);

        // Mark the Cmd+Z / Cmd+V we are about to post as ours, so the
        // daemon's own keyboard tap does not read them as the user typing
        // and cancel the pending auto-enter countdown.
        pause::begin_synthetic_input(Duration::from_millis(1_500));

        if let Err(err) = paste::replace_via_undo(&clipboard) {
            tracing::warn!(
                error = %err,
                "grammar patch replace failed; acoustic paste kept"
            );
            // Restore the anchors: the screen still holds the acoustic text.
            pause::set_last_pasted(acoustic.clone());
            return;
        }

        tracing::info!(
            stage = "grammar_patch",
            before = %acoustic,
            after = %cleaned,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "async grammar correction applied"
        );
        // Notify the UI that grammar correction silently mutated the text.
        event::global_broadcaster().grammar_corrected(acoustic.clone(), cleaned.clone());
        event::global_broadcaster().transcript_final(cleaned);
    });
}
fn print_banner(cfg: &AlwaysConfig) {
    tracing::info!(
        energy_threshold = cfg.energy_threshold,
        silence_secs = cfg.silence_secs,
        auto_enter = cfg.auto_enter,
        filter_enabled = cfg.filter_enabled,
        log_path = %cfg.log_path.display(),
        "daemon_banner"
    );
}

#[cfg(test)]
mod tests {
    // Pure-logic tests for the speech pipeline (classifier, dictation
    // merge, cooldown window, localization seam) live in
    // `super::speech_action::tests` — they were extracted alongside
    // the functions they cover so daemon orchestration and pure
    // decision logic can be exercised in isolation.

    use super::{
        AUTO_ENTER_MIN_WORDS, AlwaysConfig, GrammarOutcome, GrammarSource,
        SHORT_UTTERANCE_MAX_CHARS, SHORT_UTTERANCE_MAX_WORDS, apply_grammar_nonblocking,
        is_short_utterance, patch_is_worth_applying, should_auto_enter_for_text,
        should_refetch_speaker_model, snippet_utterance_for_log, word_sequence,
    };

    #[test]
    fn refetches_speaker_model_only_when_gate_requested_and_model_absent() {
        // The exact outage: gate on + voiceprint enrolled + model wiped
        // from the cache → must re-fetch (else it fails closed and drops
        // every utterance as "not enrolled", score -1.0).
        assert!(should_refetch_speaker_model(true, true, false));

        // Model already present → nothing to do (no wasted 26 MB fetch).
        assert!(!should_refetch_speaker_model(true, true, true));

        // Gate off, or no voiceprint enrolled → the gate is never
        // requested, so a missing model is harmless; do NOT fetch.
        assert!(!should_refetch_speaker_model(false, true, false));
        assert!(!should_refetch_speaker_model(true, false, false));
        assert!(!should_refetch_speaker_model(false, false, false));
    }

    #[test]
    fn snippet_match_redacts_utterance_when_transcript_logging_is_disabled() {
        let sensitive_utterance = "grill me about the unreleased acquisition";

        assert_eq!(
            snippet_utterance_for_log(sensitive_utterance, false),
            None,
            "snippet matches must not expose the utterance in default privacy mode"
        );
        assert_eq!(
            snippet_utterance_for_log(sensitive_utterance, true),
            Some(sensitive_utterance),
            "explicit transcript logging may include the matched utterance"
        );
    }

    #[test]
    fn short_utterance_bypass_matches_thresholds() {
        // Empty / whitespace — nothing to correct.
        assert!(is_short_utterance(""));
        assert!(is_short_utterance("   "));

        // Single short word — bypass.
        assert!(is_short_utterance("yes"));
        assert!(is_short_utterance("ok"));
        assert!(is_short_utterance("done"));
        assert!(is_short_utterance("  done  "), "trim before checking");

        // Two short words — bypass (word count threshold).
        assert!(is_short_utterance("yes please"));
        assert!(is_short_utterance("all good"));

        // Spec is `≤2 words OR ≤8 chars` (intentionally permissive) —
        // a single long word still hits the bypass via word count.
        assert!(
            is_short_utterance("acknowledged"),
            "single word always bypasses regardless of length"
        );

        // Three short words but ≤8 chars — char count gates it.
        assert!(
            is_short_utterance("a b c d"),
            "7-char 4-word input still bypasses via char threshold"
        );

        // Three words AND >8 chars — runs full pipeline.
        assert!(
            !is_short_utterance("yes no maybe"),
            "3 words above char threshold must run through correction"
        );
        assert!(
            !is_short_utterance("this is a sentence that needs cleanup"),
            "long multi-word input must run through correction"
        );

        // Sanity: thresholds are at the documented values.
        assert_eq!(SHORT_UTTERANCE_MAX_WORDS, 2);
        assert_eq!(SHORT_UTTERANCE_MAX_CHARS, 8);
    }

    #[test]
    fn auto_enter_fires_for_one_or_more_words() {
        // Empty / whitespace-only never auto-enters (nothing was said).
        assert!(!should_auto_enter_for_text(""));
        assert!(!should_auto_enter_for_text("   "));

        // A single word IS enough — the user dictates one-word commands and
        // expects Return to fire.
        assert!(should_auto_enter_for_text("hello"));
        assert!(should_auto_enter_for_text("yes"));
        assert!(should_auto_enter_for_text("hello there"));
        assert!(should_auto_enter_for_text("hello over there"));
        assert!(should_auto_enter_for_text("  hello over there  "));

        assert_eq!(AUTO_ENTER_MIN_WORDS, 1);
    }
    // ---- Grammar off the critical path ------------------------------

    #[test]
    fn grammar_source_names_are_stable_log_values() {
        // These strings land in the `latency_breakdown` log line and are
        // what any latency analysis groups on. Renaming one silently
        // breaks every historical comparison.
        assert_eq!(GrammarSource::Bypass.as_str(), "bypass");
        assert_eq!(GrammarSource::Cache.as_str(), "cache");
        assert_eq!(GrammarSource::Waited.as_str(), "waited");
        assert_eq!(GrammarSource::Deferred.as_str(), "deferred");
    }

    #[test]
    fn bypass_outcome_never_carries_a_pending_correction() {
        // Snippet expansions and short utterances take this constructor.
        // A pending correction here would let the LLM reword user-authored
        // snippet text after the fact.
        let outcome = GrammarOutcome::bypass("grill me".to_string());
        assert_eq!(outcome.text, "grill me");
        assert_eq!(outcome.source, GrammarSource::Bypass);
        assert!(outcome.pending.is_none());
    }

    #[test]
    fn word_sequence_drops_case_and_punctuation_but_keeps_apostrophes() {
        assert_eq!(
            word_sequence("Hello, there!"),
            vec!["hello".to_string(), "there".to_string()]
        );
        assert_eq!(word_sequence("hello there"), word_sequence("Hello, there."));
        // Contractions are one word, not two — otherwise "dont" vs "don't"
        // would read as a word-level change and trigger a pointless patch.
        assert_eq!(
            word_sequence("I don't"),
            vec!["i".to_string(), "don't".to_string()]
        );
    }

    #[test]
    fn patch_skips_case_only_and_punctuation_only_corrections() {
        // 9.2% of real corrections are case-only and 34.2% are
        // punctuation-or-case-only. None of those are worth taking the
        // user's text back for.
        assert!(!patch_is_worth_applying("same text", "same text"));
        assert!(!patch_is_worth_applying(
            "which is incredible.",
            "Which is incredible."
        ));
        assert!(!patch_is_worth_applying(
            "hello there how are you",
            "Hello there, how are you?"
        ));
    }

    #[test]
    fn patch_applies_when_words_actually_change() {
        // Word-level edits are the 56.6% majority and the only ones that
        // justify an undo + repaste.
        assert!(patch_is_worth_applying(
            "like I'm a cat who learned",
            "I'm a cat who learned"
        ));
        assert!(patch_is_worth_applying(
            "does it record just me speaking",
            "does it record only my speech"
        ));
    }

    #[test]
    fn grammar_is_bypassed_without_an_available_postprocessor() {
        // The local STT backend deliberately never runs the LLM
        // (`postprocess_available` requires the Groq backend), so the
        // paste path must return the acoustic text with nothing pending —
        // no spawn, no wait, no patch.
        let cfg = AlwaysConfig::default();
        assert!(!cfg.postprocess_available());
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let req = crate::always::correction_request::build("hello there friend", false);
        let acoustic = req.acoustic_text.clone();
        let outcome = apply_grammar_nonblocking(req, &cfg, rt.handle());
        assert_eq!(outcome.text, acoustic);
        assert_eq!(outcome.source, GrammarSource::Bypass);
        assert!(outcome.pending.is_none());
    }
}
