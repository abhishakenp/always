//! Speaker-embedding extraction for the "My Voice" verification gate.
//!
//! Wraps the WeSpeaker ResNet34 (VoxCeleb, English) ONNX model: 16 kHz
//! mono i16 PCM in, L2-normalised 256-dim voiceprint out. Two clips of
//! the same speaker score high cosine similarity; different speakers
//! (movie dialogue, other people in the room) score low. The daemon
//! compares each captured utterance against the enrolled voiceprint
//! and discards non-matching speech before it ever reaches STT.
//!
//! The model is 26.5 MB, so unlike Silero it is NOT `include_bytes!`d
//! into the binary — it is downloaded once (on first enrollment) into
//! the same per-user cache dir Silero materialises to, and verified
//! against a pinned sha256.
//!
//! Feature recipe matches WeSpeaker's reference `infer_onnx.py`
//! exactly — enrollment and runtime MUST share it or scores collapse:
//! waveform at int16 scale (raw cast, NO /32768), kaldi fbank with
//! 80 mel bins, 25 ms / 10 ms frames, dither 0, hamming window,
//! snip_edges, preemphasis 0.97; then per-utterance cepstral mean
//! subtraction; tensor `[1, T, 80]` → `embs [1, 256]` → L2 normalise.

use std::path::PathBuf;

use anyhow::{Context, Result};
use kaldi_fbank_rust_kautism::{FbankOptions, OnlineFbank};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use parking_lot::Mutex;

pub const MODEL_FILE: &str = "wespeaker_en_voxceleb_resnet34.onnx";
pub const MODEL_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx";
pub const MODEL_SHA256: &str = "5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94";
pub const EMBEDDING_DIM: usize = 256;

const SAMPLE_RATE: f32 = 16_000.0;
const NUM_MEL_BINS: i32 = 80;
/// Minimum audio for a usable embedding. Half a second is the floor —
/// scores get noisier below ~1s but remain far more reliable than not
/// checking at all, and the runtime gate fails CLOSED for anything
/// shorter (a snippet too short to verify is dropped, never pasted).
/// 0.3 s is the shortest window worth embedding — noisier than longer
/// voice, but the gate threshold is unchanged so a marginal early window
/// simply fails through to the next check instead of being accepted.
/// Lowered from 0.5 s so the first speaker check can fire ~200 ms sooner.
pub const MIN_EMBED_SAMPLES: usize = 4_800; // 0.3 s @ 16 kHz

/// Where the model lives on disk (`~/Library/Caches/always/` on macOS
/// — same dir Silero materialises to).
pub fn model_path() -> Result<PathBuf> {
    let dir = dirs::cache_dir().context("no cache dir")?.join("always");
    std::fs::create_dir_all(&dir)?;
    Ok(dir.join(MODEL_FILE))
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// True if the model is already on disk with the right checksum. Cheap
/// enough (~30 ms for 26 MB) to call at gate/enrollment startup.
pub fn model_present() -> bool {
    let Ok(path) = model_path() else {
        return false;
    };
    match std::fs::read(&path) {
        Ok(bytes) => sha256_hex(&bytes) == MODEL_SHA256,
        Err(_) => false,
    }
}

/// Download the model if missing or corrupt. Blocking (first
/// enrollment only); ~26 MB from GitHub releases. Writes via a temp
/// file + rename so a killed download never leaves a half-written
/// model behind.
pub fn ensure_model() -> Result<PathBuf> {
    let path = model_path()?;
    if model_present() {
        return Ok(path);
    }
    tracing::info!(url = MODEL_URL, "speaker_model_download_started");
    let bytes = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?
        .get(MODEL_URL)
        .send()
        .context("speaker model download failed")?
        .error_for_status()?
        .bytes()?;
    let got = sha256_hex(&bytes);
    if got != MODEL_SHA256 {
        anyhow::bail!("speaker model checksum mismatch: expected {MODEL_SHA256}, got {got}");
    }
    let tmp = path.with_extension("onnx.part");
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, &path)?;
    tracing::info!(bytes = bytes.len(), "speaker_model_download_done");
    Ok(path)
}

/// WeSpeaker ResNet34 speaker-embedding extractor.
///
/// `Session::run` takes `&mut self` in ort 2.0.0-rc.12, so the session
/// sits behind a `Mutex` — same shape as `SileroVad`. Inference runs
/// once per ladder window / whole-utterance check (see `coreml_cpu_ep`
/// for its cost).
pub struct SpeakerEmbedder {
    session: Mutex<Session>,
    /// Set when `session` is the ResNet trunk only (output `trunk`
    /// `[1, 256, 10, T']`): the statistics pooling and the projection run
    /// here, in f32, exactly as the full graph's tail does.
    tail: Option<PoolingTail>,
    /// The GPU trunk session (see [`coreml_gpu_ep`]) for inputs of exactly
    /// [`GPU_WINDOW_SAMPLES`] — the ladder's trailing window, ~87 % of all
    /// embeddings while media plays. Every other length stays on `session`:
    /// the ML Program re-plans for each new input shape (whole-utterance
    /// inputs of varied length cost MORE CPU there, 110.7 vs 94.5 ms).
    window: Option<Box<SpeakerEmbedder>>,
}

/// Input length routed to the GPU trunk: `vad::SPEAKER_TAIL_WINDOW_SAMPLES`.
pub(crate) const GPU_WINDOW_SAMPLES: usize = 24_000;
/// Fbank frames of a [`GPU_WINDOW_SAMPLES`] input (25 ms / 10 ms, snip edges).
const GPU_WINDOW_FRAMES: usize = 1 + (GPU_WINDOW_SAMPLES - 400) / 160;

/// The full graph's tail after the last ResNet block: mean and unbiased
/// standard deviation (+1e-7 under the root) over time, concatenated
/// `[mean(2560); std(2560)]`, `Gemm` to 256, minus `mean_vec`.
pub(crate) struct PoolingTail {
    /// `[256 * 5120]` row-major (`transB` = 1: one row per output).
    weight: Vec<f32>,
    bias: Vec<f32>,
    mean_vec: Vec<f32>,
}

const POOL_CHANNELS: usize = 2560;

impl PoolingTail {
    pub(crate) fn from_le_bytes(bytes: &[u8]) -> Result<Self> {
        let want = (EMBEDDING_DIM * 2 * POOL_CHANNELS + 2 * EMBEDDING_DIM) * 4;
        anyhow::ensure!(bytes.len() == want, "pooling tail: {} bytes, want {want}", bytes.len());
        let v: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let w = EMBEDDING_DIM * 2 * POOL_CHANNELS;
        Ok(Self {
            weight: v[..w].to_vec(),
            bias: v[w..w + EMBEDDING_DIM].to_vec(),
            mean_vec: v[w + EMBEDDING_DIM..].to_vec(),
        })
    }

    /// `h`: trunk output `[2560 * t]` (channel-major, time innermost).
    fn apply(&self, h: &[f32], t: usize) -> Result<Vec<f32>> {
        anyhow::ensure!(t >= 2 && h.len() == POOL_CHANNELS * t, "trunk output {} for t={t}", h.len());
        let n = t as f32;
        let mut z = vec![0f32; 2 * POOL_CHANNELS];
        for (c, row) in h.chunks_exact(t).enumerate() {
            let mean = row.iter().sum::<f32>() / n;
            let var = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
            z[c] = mean;
            z[POOL_CHANNELS + c] = (var * n / (n - 1.0) + 1e-7).sqrt();
        }
        Ok(self
            .weight
            .chunks_exact(2 * POOL_CHANNELS)
            .zip(self.bias.iter().zip(&self.mean_vec))
            .map(|(w, (b, m))| w.iter().zip(&z).map(|(a, x)| a * x).sum::<f32>() + b - m)
            .collect())
    }
}

/// ORT's own CPU kernels, one thread — the original speaker session.
fn cpu_session_builder() -> Result<ort::session::builder::SessionBuilder> {
    // rc.12 builder errors carry non-Send/Sync payloads and won't
    // convert to anyhow via `?` — normalise to plain `ort::Error`
    // first (same workaround vad-rs uses).
    Ok(Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| -> ort::Error { e.into() })?
        .with_intra_threads(1)
        .map_err(|e| -> ort::Error { e.into() })?
        .with_inter_threads(1)
        .map_err(|e| -> ort::Error { e.into() })?)
}

/// The CPU session: CoreML's CPU backend (fp32, Accelerate), not ORT's CPU
/// kernels: the same embedding (cosine to ORT's >= 0.9999988, scores within
/// 7.5e-7) for a sixth to a ninth of the CPU (`embed_cost_probe`,
/// 2026-10-17). Since 2026-10-19 it is the FALLBACK and the reference the
/// GPU session is checked against at load (see [`SpeakerEmbedder::new`]).
///
/// - `NeuralNetwork`, not `MLProgram`: ORT's MLProgram path returns NaN for
///   the whole graph (its std-pooling tail) — the GPU session runs the
///   trunk only for that reason.
/// - Not the NeuralNetwork GPU/ANE paths: those run fp16 and move scores by
///   up to ~3e-3 (measured 2.8e-3 on 547 windows, 2026-10-19).
/// - A non-finite embedding is an error: the speaker is not verified.
/// - If CoreML cannot be registered at all, ORT itself runs its CPU kernels.
fn coreml_cpu_ep() -> ort::ep::ExecutionProviderDispatch {
    use ort::ep::coreml::{ComputeUnits, ModelFormat};
    ort::ep::CoreML::default()
        .with_model_format(ModelFormat::NeuralNetwork)
        .with_compute_units(ComputeUnits::CPUOnly)
        .build()
}

/// The trunk as an ML Program on the GPU, fp32 (low-precision GPU
/// accumulation off): per 1.5 s ladder window 5.0 ms process CPU against
/// the CPU session's 40 ms under the same background QoS (`embed_cost_probe`,
/// ser-18 scene, 547 windows, 2026-10-19), scores within 8.1e-7 of it.
fn coreml_gpu_ep() -> ort::ep::ExecutionProviderDispatch {
    use ort::ep::coreml::{ComputeUnits, ModelFormat};
    ort::ep::CoreML::default()
        .with_model_format(ModelFormat::MLProgram)
        .with_compute_units(ComputeUnits::CPUAndGPU)
        .with_low_precision_accumulation_on_gpu(false)
        .build()
}

/// Largest L2 distance between the GPU and CPU embeddings of the load-time
/// probes for the GPU session to be used: fp32 agreement is ~4e-6, an fp16
/// path ~1e-2, so this separates them with room on both sides.
const GPU_AGREEMENT_MAX_L2: f32 = 1e-4;

/// Deterministic voice-like probes for the load-time agreement check
/// (harmonics on a wavering pitch + noise; two window-length voices).
fn agreement_probes() -> Vec<Vec<i16>> {
    [(GPU_WINDOW_SAMPLES, 140.0f32, 1u32), (GPU_WINDOW_SAMPLES, 210.0, 7)]
        .iter()
        .map(|&(n, f0, seed)| {
            let mut rng = seed.wrapping_mul(2_654_435_761);
            let mut phase = 0f32;
            (0..n)
                .map(|i| {
                    let t = i as f32 / SAMPLE_RATE;
                    phase += std::f32::consts::TAU * f0 * (1.0 + 0.08 * (t * 3.1).sin()) / SAMPLE_RATE;
                    let voiced: f32 = (1..8).map(|h| (phase * h as f32).sin() / h as f32).sum();
                    rng ^= rng << 13;
                    rng ^= rng >> 17;
                    rng ^= rng << 5;
                    let noise = (rng as f32 / u32::MAX as f32) - 0.5;
                    ((voiced * (0.6 + 0.4 * (t * 4.0).sin().abs()) + 0.1 * noise) * 6_000.0) as i16
                })
                .collect()
        })
        .collect()
}

impl SpeakerEmbedder {
    /// Load the model from the cache dir. Errors if it hasn't been
    /// downloaded yet — call `ensure_model()` first on enrollment
    /// paths; the runtime gate treats a missing model as "gate off".
    ///
    /// The GPU trunk session takes the ladder windows only when its
    /// embeddings of [`agreement_probes`] match the CPU session's within
    /// [`GPU_AGREEMENT_MAX_L2`]; otherwise (no GPU, a CoreML change, any
    /// load error) the CPU session runs everything, as before. Either way
    /// the gate sees the same scores.
    pub fn new() -> Result<Self> {
        let path = model_path()?;
        let mut cpu = Self::cpu(&path)?;
        let started = std::time::Instant::now();
        match Self::gpu_trunk(&path).and_then(|gpu| {
            let mut worst = 0f32;
            for probe in agreement_probes() {
                let (a, b) = (gpu.embed(&probe)?, cpu.embed(&probe)?);
                let l2 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
                worst = worst.max(l2);
            }
            Ok((gpu, worst))
        }) {
            Ok((gpu, l2)) if l2 <= GPU_AGREEMENT_MAX_L2 => {
                tracing::info!(
                    backend = "coreml-gpu-trunk",
                    agreement_l2 = l2,
                    check_ms = started.elapsed().as_millis() as u64,
                    "speaker_model_backend"
                );
                cpu.window = Some(Box::new(gpu));
                Ok(cpu)
            }
            Ok((_, l2)) => {
                tracing::warn!(backend = "coreml-cpu", agreement_l2 = l2, "speaker_model_backend");
                Ok(cpu)
            }
            Err(e) => {
                tracing::warn!(backend = "coreml-cpu", error = %e, "speaker_model_backend");
                Ok(cpu)
            }
        }
    }

    fn cpu(path: &std::path::Path) -> Result<Self> {
        let session = cpu_session_builder()?
            .with_execution_providers([coreml_cpu_ep()])
            .map_err(|e| -> ort::Error { e.into() })?
            .commit_from_file(path)
            .with_context(|| format!("loading speaker model from {}", path.display()))?;
        Ok(Self { session: Mutex::new(session), tail: None, window: None })
    }

    fn gpu_trunk(path: &std::path::Path) -> Result<Self> {
        let split = crate::always::onnx_trunk::split(&std::fs::read(path)?)?;
        let tail = PoolingTail::from_le_bytes(&split.tail)?;
        let session = cpu_session_builder()?
            .with_execution_providers([coreml_gpu_ep().error_on_failure()])
            .map_err(|e| -> ort::Error { e.into() })?
            .commit_from_memory(&split.model)
            .context("loading the speaker trunk")?;
        Ok(Self { session: Mutex::new(session), tail: Some(tail), window: None })
    }

    /// Run the model once on `session`; the raw (un-normalised) output.
    fn run(session: &mut Session, tail: Option<&PoolingTail>, feats: Vec<f32>) -> Result<Vec<f32>> {
        let num_frames = feats.len() / NUM_MEL_BINS as usize;
        let tensor = ndarray::Array3::from_shape_vec((1, num_frames, NUM_MEL_BINS as usize), feats)
            .context("fbank tensor shape")?;
        let value = Value::from_array(tensor)?;
        let outputs = session.run(ort::inputs!["feats" => value])?;
        let Some(tail) = tail else {
            let (_, raw) = outputs
                .get("embs")
                .context("model output `embs` missing")?
                .try_extract_tensor::<f32>()?;
            return Ok(raw.to_vec());
        };
        let (shape, h) = outputs
            .get(crate::always::onnx_trunk::TRUNK_OUTPUT)
            .context("trunk output missing")?
            .try_extract_tensor::<f32>()?;
        let t = *shape.last().context("trunk shape")? as usize;
        tail.apply(h, t)
    }

    /// Compute the L2-normalised 256-dim embedding of an utterance.
    /// `samples`: 16 kHz mono i16 PCM, at least `MIN_EMBED_SAMPLES`.
    pub fn embed(&self, samples: &[i16]) -> Result<Vec<f32>> {
        if samples.len() < MIN_EMBED_SAMPLES {
            anyhow::bail!(
                "utterance too short for speaker embedding: {} samples (< {MIN_EMBED_SAMPLES})",
                samples.len()
            );
        }
        self.embed_feats(compute_fbank_cmn(samples)?)
    }

    /// The model half of [`Self::embed`]: CMN'd fbank `[T * 80]` → embedding.
    fn embed_feats(&self, feats: Vec<f32>) -> Result<Vec<f32>> {
        if let Some(window) = &self.window
            && feats.len() == GPU_WINDOW_FRAMES * NUM_MEL_BINS as usize
        {
            return window.embed_feats(feats);
        }
        let raw = Self::run(&mut self.session.lock(), self.tail.as_ref(), feats)?;
        // Never seen on this backend (0 of 14,900 windows, 2026-10-17); if it
        // ever happens the speaker is simply not verified (the gate rejects).
        if raw.iter().any(|v| !v.is_finite()) {
            anyhow::bail!("speaker embedding is not finite");
        }
        if raw.len() != EMBEDDING_DIM {
            anyhow::bail!(
                "unexpected embedding size {} (want {EMBEDDING_DIM})",
                raw.len()
            );
        }
        Ok(l2_normalize(&raw))
    }
}

/// Kaldi 80-bin log-mel fbank (WeSpeaker recipe) + per-utterance CMN.
/// Returns a flat `[T * 80]` row-major matrix.
fn compute_fbank_cmn(samples: &[i16]) -> Result<Vec<f32>> {
    let mut opts = FbankOptions::default();
    opts.frame_opts.samp_freq = SAMPLE_RATE;
    opts.frame_opts.dither = 0.0;
    // WeSpeaker trains with torchaudio's hamming window, not kaldi's
    // default povey. The pointer must outlive the C++ object, hence a
    // C-string literal (static lifetime).
    opts.frame_opts.window_type = c"hamming".as_ptr();
    opts.mel_opts.num_bins = NUM_MEL_BINS;
    let mut fbank = OnlineFbank::new(opts);

    // WeSpeaker feeds waveforms at int16 scale — cast, do NOT divide.
    let wave: Vec<f32> = samples.iter().map(|&s| s as f32).collect();
    fbank.accept_waveform(SAMPLE_RATE, &wave);
    fbank.input_finished();

    let num_frames = fbank.num_ready_frames();
    if num_frames <= 0 {
        anyhow::bail!("no fbank frames produced ({} samples)", samples.len());
    }
    let bins = NUM_MEL_BINS as usize;
    let mut feats = Vec::with_capacity(num_frames as usize * bins);
    for i in 0..num_frames {
        let frame = fbank.get_frame(i).context("fbank frame ready-count lied")?;
        feats.extend_from_slice(frame);
    }

    // Per-utterance CMN: subtract each mel bin's mean across frames.
    let n = num_frames as f32;
    let mut means = vec![0f32; bins];
    for frame in feats.chunks_exact(bins) {
        for (m, v) in means.iter_mut().zip(frame) {
            *m += v;
        }
    }
    for m in &mut means {
        *m /= n;
    }
    for frame in feats.chunks_exact_mut(bins) {
        for (v, m) in frame.iter_mut().zip(&means) {
            *v -= m;
        }
    }
    Ok(feats)
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
    v.iter().map(|x| x / norm).collect()
}

/// Cosine similarity of two L2-normalised embeddings (= dot product).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Process-wide embedder, loaded lazily on first use. Load failures
/// are NOT cached — the model may appear later (first enrollment
/// downloads it), so a missing model this utterance can succeed the
/// next. The `model_present()` fast-path keeps the retry cheap when
/// the file simply isn't there.
static GLOBAL: Mutex<Option<std::sync::Arc<SpeakerEmbedder>>> = Mutex::new(None);

pub fn global() -> Option<std::sync::Arc<SpeakerEmbedder>> {
    let mut slot = GLOBAL.lock();
    if slot.is_none() {
        if !model_present() {
            return None;
        }
        match SpeakerEmbedder::new() {
            Ok(e) => *slot = Some(std::sync::Arc::new(e)),
            Err(e) => {
                tracing::warn!(error = %e, "speaker_embedder_load_failed");
                return None;
            }
        }
    }
    slot.clone()
}

/// Combine per-step enrollment embeddings into one voiceprint:
/// element-wise sum, then L2-normalise (sherpa-onnx's
/// `SpeakerEmbeddingManager` convention — equivalent to the mean).
pub fn combine_embeddings(embeddings: &[Vec<f32>]) -> Result<Vec<f32>> {
    let first = embeddings.first().context("no embeddings to combine")?;
    let dim = first.len();
    let mut sum = vec![0f32; dim];
    for e in embeddings {
        if e.len() != dim {
            anyhow::bail!("embedding dim mismatch: {} vs {dim}", e.len());
        }
        for (s, v) in sum.iter_mut().zip(e) {
            *s += v;
        }
    }
    Ok(l2_normalize(&sum))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_voice(seed: u64, secs: f32) -> Vec<i16> {
        // Deterministic pseudo-speech: a fundamental + formant-ish
        // harmonics with slow pitch wobble, distinct per seed. Not real
        // speech, but distinct "speakers" produce distinct spectra.
        let n = (SAMPLE_RATE * secs) as usize;
        let f0 = 90.0 + (seed % 7) as f32 * 25.0;
        let f1 = 400.0 + (seed % 5) as f32 * 130.0;
        let f2 = 1200.0 + (seed % 3) as f32 * 300.0;
        (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE;
                let wobble = 1.0 + 0.02 * (2.0 * std::f32::consts::PI * 2.3 * t).sin();
                let s = 0.5 * (2.0 * std::f32::consts::PI * f0 * wobble * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * f1 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * f2 * t).sin();
                (s * 8000.0) as i16
            })
            .collect()
    }

    #[test]
    fn cosine_of_identical_normalized_vectors_is_one() {
        let v = l2_normalize(&[1.0, 2.0, 3.0]);
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn combine_embeddings_is_normalized() {
        let a = l2_normalize(&[1.0, 0.0]);
        let b = l2_normalize(&[0.0, 1.0]);
        let c = combine_embeddings(&[a, b]).unwrap();
        let norm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn fbank_produces_expected_frame_count() {
        // 1 s @ 16 kHz, snip_edges: 1 + (16000-400)/160 = 98 frames.
        let samples = synth_voice(1, 1.0);
        let feats = compute_fbank_cmn(&samples).unwrap();
        assert_eq!(feats.len(), 98 * 80);
        // CMN: per-bin means should now be ~0.
        let chunks = feats.as_chunks::<80>();
        let mean0: f32 = chunks.0.iter().map(|f| f[0]).sum::<f32>() / 98.0;
        assert!(mean0.abs() < 1e-3, "CMN should zero bin means, got {mean0}");
    }

    // The tests below need the 26.5 MB model on disk; skip cleanly
    // when absent so CI without the download still passes.
    fn embedder_or_skip() -> Option<SpeakerEmbedder> {
        if !model_present() {
            return None;
        }
        Some(SpeakerEmbedder::new().expect("model loads"))
    }

    #[test]
    fn same_clip_scores_near_one() {
        let Some(e) = embedder_or_skip() else { return };
        let clip = synth_voice(1, 3.0);
        let a = e.embed(&clip).unwrap();
        let b = e.embed(&clip).unwrap();
        assert!(
            cosine(&a, &b) > 0.999,
            "identical audio must embed identically, got {}",
            cosine(&a, &b)
        );
    }

    #[test]
    fn different_synthetic_speakers_score_below_same_speaker() {
        let Some(e) = embedder_or_skip() else { return };
        // Same "speaker", two different clips (different duration).
        let same_a = e.embed(&synth_voice(1, 3.0)).unwrap();
        let same_b = e.embed(&synth_voice(1, 4.0)).unwrap();
        // Different "speaker".
        let other = e.embed(&synth_voice(4, 3.0)).unwrap();
        let same_score = cosine(&same_a, &same_b);
        let cross_score = cosine(&same_a, &other);
        assert!(
            same_score > cross_score,
            "same-source clips ({same_score}) must outscore cross-source ({cross_score})"
        );
    }

    /// Per-window cost probe (offline, no mic). `SPK_PROBE_WAVS`: 16 kHz
    /// mono s16 WAVs, comma-separated. `SPK_PROBE_MODE=ladder` (default):
    /// the ladder's 1.5 s trailing windows at its 0.25 s step; `whole`: the
    /// whole-utterance inputs (last <= 6 s of each file) plus every prefix
    /// length from 0.3 s in 0.23 s steps (varied T). Measures the shipped
    /// session (`SpeakerEmbedder::new`; `SPK_PROBE_EP` names it in the output). Times fbank and model separately
    /// (wall + process CPU) and writes every embedding to `SPK_PROBE_OUT`
    /// (raw f32 LE, 256 per window) for a window-by-window comparison.
    #[test]
    #[ignore = "offline cost probe; needs SPK_PROBE_* env + model"]
    fn embed_cost_probe() {
        let Ok(wavs) = std::env::var("SPK_PROBE_WAVS") else {
            return;
        };
        let ep = std::env::var("SPK_PROBE_EP").unwrap_or_else(|_| "shipped".into());
        let out = std::env::var("SPK_PROBE_OUT").expect("SPK_PROBE_OUT");
        let whole = std::env::var("SPK_PROBE_MODE").is_ok_and(|m| m == "whole");
        let cpu_s = || {
            let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
            unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
            let tv = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1e6;
            tv(ru.ru_utime) + tv(ru.ru_stime)
        };
        let (lc, lt) = (cpu_s(), std::time::Instant::now());
        let e = probe_embedder(&ep);
        let _ = e.embed(&synth_voice(1, 1.5));
        eprintln!(
            "probe ep={ep} load+first-run wall={:.0} ms cpu={:.0} ms",
            lt.elapsed().as_secs_f64() * 1e3,
            (cpu_s() - lc) * 1e3
        );
        let mut bytes = Vec::new();
        let (mut n, mut fb_wall, mut fb_cpu, mut m_wall, mut m_cpu) =
            (0usize, 0f64, 0f64, 0f64, 0f64);
        let mut m_walls = Vec::new();
        let mut secs = 0f64;
        for wav in wavs.split(',') {
            let pcm = read_wav_s16(wav);
            let mut wins: Vec<&[i16]> = Vec::new();
            if whole {
                if pcm.len() >= MIN_EMBED_SAMPLES {
                    wins.push(&pcm[pcm.len() - pcm.len().min(96_000)..]);
                }
                let mut l = MIN_EMBED_SAMPLES;
                while l <= pcm.len().min(96_000) {
                    wins.push(&pcm[..l]);
                    l += 3_680;
                }
            } else {
                let mut end = 24_000;
                while end <= pcm.len() {
                    wins.push(&pcm[end - 24_000..end]);
                    end += 4_000;
                }
            }
            for w in wins {
                let (c0, t0) = (cpu_s(), std::time::Instant::now());
                let feats = compute_fbank_cmn(w).unwrap();
                let (c1, t1) = (cpu_s(), std::time::Instant::now());
                let emb = e.embed_feats(feats).unwrap();
                let (c2, t2) = (cpu_s(), std::time::Instant::now());
                fb_wall += (t1 - t0).as_secs_f64();
                fb_cpu += c1 - c0;
                m_wall += (t2 - t1).as_secs_f64();
                m_cpu += c2 - c1;
                m_walls.push((t2 - t1).as_secs_f64());
                secs += w.len() as f64 / 16_000.0;
                for v in &emb {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                n += 1;
            }
        }
        std::fs::write(&out, &bytes).unwrap();
        m_walls.sort_by(f64::total_cmp);
        let p = |q: f64| m_walls[((m_walls.len() - 1) as f64 * q) as usize] * 1e3;
        eprintln!(
            "probe ep={ep} mode={} windows={n} audio_s/win={:.2} | fbank ms/win wall={:.2} cpu={:.2} | model ms/win wall={:.2} cpu={:.2} p50={:.1} p95={:.1} max={:.1}",
            if whole { "whole" } else { "ladder" },
            secs / n as f64,
            fb_wall / n as f64 * 1e3,
            fb_cpu / n as f64 * 1e3,
            m_wall / n as f64 * 1e3,
            m_cpu / n as f64 * 1e3,
            p(0.5),
            p(0.95),
            p(1.0),
        );
    }

    fn read_wav_s16(path: &str) -> Vec<i16> {
        let b = std::fs::read(path).unwrap();
        // Walk RIFF chunks to "data" (16 kHz mono s16 asserted by "fmt ").
        let mut i = 12;
        loop {
            let id = &b[i..i + 4];
            let len = u32::from_le_bytes(b[i + 4..i + 8].try_into().unwrap()) as usize;
            if id == b"fmt " {
                let ch = u16::from_le_bytes(b[i + 10..i + 12].try_into().unwrap());
                let sr = u32::from_le_bytes(b[i + 12..i + 16].try_into().unwrap());
                let bits = u16::from_le_bytes(b[i + 22..i + 24].try_into().unwrap());
                assert_eq!((ch, sr, bits), (1, 16_000, 16), "{path}");
            }
            if id == b"data" {
                let d = &b[i + 8..(i + 8 + len).min(b.len())];
                return d
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| i16::from_le_bytes(*c))
                    .collect();
            }
            i += 8 + len + (len & 1);
        }
    }

    fn probe_embedder(ep: &str) -> SpeakerEmbedder {
        use ort::ep::coreml::{ComputeUnits, ModelFormat};
        let ep_name = ep;
        let (fmt, units) = match ep {
            "shipped" => return SpeakerEmbedder::new().unwrap(),
            "cpu" => return SpeakerEmbedder::cpu(&model_path().unwrap()).unwrap(),
            "gpu" => return SpeakerEmbedder::gpu_trunk(&model_path().unwrap()).unwrap(),
            "ort-cpu" => (ModelFormat::NeuralNetwork, ComputeUnits::CPUOnly),
            "nn-gpu" => (ModelFormat::NeuralNetwork, ComputeUnits::CPUAndGPU),
            "nn-all" => (ModelFormat::NeuralNetwork, ComputeUnits::All),
            "nn-ane" => (ModelFormat::NeuralNetwork, ComputeUnits::CPUAndNeuralEngine),
            "ml-cpu" => (ModelFormat::MLProgram, ComputeUnits::CPUOnly),
            "ml-gpu" => (ModelFormat::MLProgram, ComputeUnits::CPUAndGPU),
            "ml-ane" => (ModelFormat::MLProgram, ComputeUnits::CPUAndNeuralEngine),
            other => panic!("SPK_PROBE_EP {other}"),
        };
        let ep = ort::ep::CoreML::default()
            .with_model_format(fmt)
            .with_compute_units(units)
            .with_profile_compute_plan(std::env::var_os("SPK_PROBE_PLAN").is_some())
            .build();
        // SPK_PROBE_TRUNK=<trunk.onnx>,<tail.bin>: the trunk-only graph.
        let trunk = std::env::var("SPK_PROBE_TRUNK").ok();
        let (path, tail) = match &trunk {
            Some(t) => {
                let (m, w) = t.split_once(',').unwrap();
                (PathBuf::from(m), Some(PoolingTail::from_le_bytes(&std::fs::read(w).unwrap()).unwrap()))
            }
            None => (model_path().unwrap(), None),
        };
        let mut b = cpu_session_builder().unwrap();
        if ep_name != "ort-cpu" {
            b = b.with_execution_providers([ep]).map_err(|e| -> ort::Error { e.into() }).unwrap();
        }
        let session = b.commit_from_file(path).unwrap();
        SpeakerEmbedder { session: Mutex::new(session), tail, window: None }
    }

    fn footprint_mb() -> f64 {
        let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
        let r = unsafe {
            libc::proc_pid_rusage(
                std::process::id() as i32,
                libc::RUSAGE_INFO_V2,
                (&mut info as *mut libc::rusage_info_v2).cast(),
            )
        };
        assert_eq!(r, 0);
        info.ri_phys_footprint as f64 / 1048576.0
    }

    /// Process footprint of each session kind after 50 window embeds:
    /// `SPK_MEM=cpu|gpu|both cargo test --release --lib speaker_mem -- --ignored --nocapture`.
    #[test]
    #[ignore = "memory probe; one session kind per process"]
    fn speaker_mem() {
        let Ok(kind) = std::env::var("SPK_MEM") else { return };
        let base = footprint_mb();
        let path = model_path().unwrap();
        let e = match kind.as_str() {
            "cpu" => SpeakerEmbedder::cpu(&path).unwrap(),
            "gpu" => SpeakerEmbedder::gpu_trunk(&path).unwrap(),
            _ => SpeakerEmbedder::new().unwrap(),
        };
        let loaded = footprint_mb();
        for i in 0..50 {
            e.embed(&synth_voice(i, 1.5)).unwrap();
        }
        e.embed(&synth_voice(1, 3.7)).unwrap();
        unsafe extern "C" {
            fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
        }
        unsafe { malloc_zone_pressure_relief(std::ptr::null_mut(), 0) };
        if std::env::var_os("SPK_MEM_HOLD").is_some() {
            eprintln!("holding pid {}", std::process::id());
            std::thread::sleep(std::time::Duration::from_secs(std::env::var("SPK_MEM_HOLD").ok().and_then(|s| s.parse().ok()).unwrap_or(8)));
        }
        eprintln!("speaker_mem {kind}: base {base:.1} MB, loaded +{:.1} MB, after embeds +{:.1} MB", loaded - base, footprint_mb() - base);
    }

    /// The in-memory split (trunk graph + Rust pooling tail) computes the
    /// full graph's embedding: both on ORT's own CPU kernels (no CoreML),
    /// then the shipped GPU trunk against the CPU session.
    #[test]
    fn trunk_split_matches_full_graph() {
        let Ok(path) = model_path() else { return };
        if !path.exists() {
            return;
        }
        let split = crate::always::onnx_trunk::split(&std::fs::read(&path).unwrap()).unwrap();
        let trunk = SpeakerEmbedder {
            session: Mutex::new(cpu_session_builder().unwrap().commit_from_memory(&split.model).unwrap()),
            tail: Some(PoolingTail::from_le_bytes(&split.tail).unwrap()),
            window: None,
        };
        let full = SpeakerEmbedder {
            session: Mutex::new(cpu_session_builder().unwrap().commit_from_file(&path).unwrap()),
            tail: None,
            window: None,
        };
        let cpu = SpeakerEmbedder::cpu(&path).unwrap();
        let gpu = SpeakerEmbedder::gpu_trunk(&path).unwrap();
        let l2 = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
        for secs in [0.3f32, 1.5, 4.0] {
            let v = synth_voice(3, secs);
            let d = l2(&trunk.embed(&v).unwrap(), &full.embed(&v).unwrap());
            assert!(d < 1e-5, "split vs full graph at {secs}s: {d}");
            let g = l2(&gpu.embed(&v).unwrap(), &cpu.embed(&v).unwrap());
            assert!(g < GPU_AGREEMENT_MAX_L2, "gpu trunk vs cpu at {secs}s: {g}");
        }
    }

    #[test]
    fn too_short_utterance_errors() {
        let Some(e) = embedder_or_skip() else { return };
        // Just under MIN_EMBED_SAMPLES (0.3 s since the floor was lowered
        // from 0.5 s — this test still used 0.3 s and had gone stale).
        let short = synth_voice(1, 0.25);
        assert!(short.len() < MIN_EMBED_SAMPLES);
        assert!(e.embed(&short).is_err());
    }

    /// Real-speech sanity check with macOS TTS voices. Needs `say` +
    /// `sox` + the downloaded model, so it's ignored by default — run
    /// with `cargo test speaker_embed -- --ignored`.
    #[test]
    #[ignore]
    fn real_tts_voices_separate_cleanly() {
        let Some(e) = embedder_or_skip() else { return };
        let tts = |voice: &str, text: &str| -> Vec<i16> {
            let dir = std::env::temp_dir();
            let aiff = dir.join(format!("always-spk-test-{voice}-{}.aiff", text.len()));
            let raw = aiff.with_extension("raw");
            let ok = std::process::Command::new("say")
                .args(["-v", voice, "-o"])
                .arg(&aiff)
                .arg(text)
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            assert!(ok, "say -v {voice} failed");
            let ok = std::process::Command::new("/opt/homebrew/bin/sox")
                .arg(&aiff)
                .args([
                    "-r",
                    "16000",
                    "-c",
                    "1",
                    "-b",
                    "16",
                    "-e",
                    "signed-integer",
                    "-t",
                    "raw",
                ])
                .arg(&raw)
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            assert!(ok, "sox convert failed");
            let bytes = std::fs::read(&raw).unwrap();
            let _ = std::fs::remove_file(&aiff);
            let _ = std::fs::remove_file(&raw);
            let chunks = bytes.as_chunks::<2>();
            chunks
                .0
                .iter()
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect()
        };

        let sam_a = e
            .embed(&tts(
                "Samantha",
                "The quick brown fox jumps over the lazy dog near the riverbank.",
            ))
            .unwrap();
        let sam_b = e
            .embed(&tts(
                "Samantha",
                "Yesterday I walked to the market and bought fresh bread and coffee.",
            ))
            .unwrap();
        let dan = e
            .embed(&tts(
                "Daniel",
                "The quick brown fox jumps over the lazy dog near the riverbank.",
            ))
            .unwrap();

        let same = cosine(&sam_a, &sam_b);
        let cross = cosine(&sam_a, &dan);
        assert!(
            same > 0.5,
            "same TTS voice, different sentences should exceed gate threshold, got {same}"
        );
        assert!(
            cross < same - 0.15,
            "different TTS voices should score clearly lower (same={same}, cross={cross})"
        );
    }
}
