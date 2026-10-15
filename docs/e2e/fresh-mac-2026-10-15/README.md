# Fresh-Mac first-run test — 2026-10-15

Environment: GitHub-hosted `macos-26` runner (macOS 26.6.2, Apple M1 virtual,
3 cores, 7 GB), driven over SSH. Homebrew hidden (`/opt/homebrew` moved away),
so no SoX, no Homebrew, no API key, no prior TCC grants, empty prefs.
The VM has no microphone: speech is synthesized with `say` (Samantha voice)
and played into a BlackHole 2ch loopback that is the default input
(`harness/playdev.swift` plays to the named device, so the system output and
the echo canceller never see it). Text is read back from TextEdit through
the Accessibility API every 20 ms (`harness/watchtext.swift`).

Builds (all `cargo build --release` + `Always/build.sh`, ad-hoc signed):

| label | source |
|---|---|
| upstream | LivioGama/always `main` @ 9246b50 |
| pr103 | abhishakenp/always `release/v0.0.2` @ 6538519 (PR #103 head) |
| final | `fix/first-run-new-user` (this fix) |

Harness-only shortcuts, not app behaviour: the Accessibility toggle in System
Settings needs the admin password, so `harness/grant_ax.sh` writes the TCC row
and the app is relaunched; the harness's own SSH session was given screen
capture / AppleEvents / accessibility. Every app-level prompt (Microphone,
Speech Recognition) was answered by clicking the real system dialog.

## Results (`results/*.summary.json`)

| build | typed | WER | end of speech → text in TextEdit |
|---|---|---|---|
| upstream | 0/3 | 100% | never (`transcriber_init_failed`: no GROQ_API_KEY; enrolment: "Install SoX"; TextEdit also not on the allowlist) |
| pr103 (TextEdit allowlisted) | 0/10 | 100% | never (cpal: "No compatible mono input config" on the stereo input; enrolment: "Install SoX") |
| final | 9/10 | 15.6% (4.7% on the 9 typed) | median 2.08 s; 1.22 s when the transcript ends in punctuation |
| fix, earlier build (same capture/engine code, before punctuation) | 10/10 | 3.1% | median 2.10 s |
| final, different voice (Daniel) after enrolling Samantha | 0/3 | — | rejected by My Voice (scores 0.13–0.23 < 0.50), as intended |

Live preview (first words on screen, `live_preview_first_text.since_onset_ms`):
97–419 ms, median 215 ms.

The missed sentence: the speculative transcript was right, but with no final
period the endpoint waited the extended window, the utterance crossed the 6 s
chunk boundary, and the chunk re-transcription came back empty — see the PR
comment's open items.

Screenshots: `screenshots/up-*` (upstream), `pr-*` (PR #103), `final-*` (fix).
