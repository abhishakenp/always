// always-capture — microphone recorder for the Always daemon, a drop-in
// replacement for `rec` (SoX): 16 kHz mono signed 16-bit little-endian PCM
// on stdout, diagnostics on stderr, one process for the daemon's lifetime.
//
// What it adds over `rec`: Apple's voice-processing I/O (VPIO) acoustic echo
// cancellation, so audio the Mac itself is playing (a YouTube video, music)
// is subtracted from the microphone signal BEFORE the daemon's VAD, speaker
// gate and speech recognizers ever see it.
//
// Echo cancellation only costs CPU when there is an echo to cancel: while no
// other process on the Mac is playing audio (and no status cue is rendering)
// the voice-processing DSP is BYPASSED, and it is re-engaged the moment any
// process starts output — event-driven via CoreAudio process objects, no
// polling. Measured 2026-10-14 on the user's MacBook Air (60 s windows, same
// room): VPIO active 15.1 CPU-s/min, VPIO bypassed 3.0 CPU-s/min.
//
// 2026-10-15: default path is now TAP mode (see "Echo cancellation against
// what the Mac is PLAYING" below) — VPIO turned out not to cancel other
// processes' audio at all, and it alters system output — so without the
// permission the mic is captured plain (VPIO only with --vp).
//
// Usage: always-capture [--no-vp] [--vp] [--tap] [--aec-always] [--agc] [--duck]
//   --vp / --tap  force the VPIO / tap-reference canceller (env ALWAYS_CAPTURE_AEC)
//   --tap-status  print the System Audio Recording permission and exit
//   --request-tap-permission  show the permission prompt and exit
//   --ref-only    diagnostics: write the 16 kHz tap reference (no mic) to stdout
//   --no-vp  plain capture (no echo cancellation) — measurement baseline
//   --aec-always  never bypass the echo canceller (pre-2026-10-14 behaviour)
//   --agc    keep VPIO's automatic gain control on (default off: the daemon's
//            energy thresholds are calibrated on un-gained mic levels)
//   --duck   let VPIO duck other apps' audio (default: no ducking — Always must
//            never make the user's video quieter)
// Exits when stdout closes (EPIPE) or the parent process dies.

import Accelerate
import AVFoundation
import CoreAudio
import Foundation

signal(SIGPIPE, SIG_IGN)
let args = Set(CommandLine.arguments.dropFirst())
/// Live capture is latency-critical: without this activity the system may
/// coalesce and defer this process's work once the display sleeps (App
/// Nap / timer coalescing), and the mic audio then reaches the daemon in
/// bursts — 2026-10-19 05:54:16 the display went off and 8.9 s of audio
/// arrived at once; 5.6 s and 12.7 s later with the display still off.
/// Idle system sleep stays allowed; it never keeps the display awake.
let captureActivity = ProcessInfo.processInfo.beginActivity(
    options: [.userInitiatedAllowingIdleSystemSleep, .latencyCritical],
    reason: "live microphone capture for dictation")
let parentPid = getppid()

func log(_ s: String) {
    // fputs, not FileHandle.write: the latter raises (and abort()s) on EPIPE
    // when the daemon has gone (same crash as Iris's pcm-play/brief-fm).
    _ = "always-capture: \(s)\n".withCString { fputs($0, stderr) }
}

// ---- Echo cancellation against what the Mac is PLAYING ----------------------
// Measured 2026-10-15 (prod logs, built-in speakers): with Apple's voice-
// processing unit "engaged" for a YouTube/Music tab (17:11–17:38Z) the song's
// lyrics were transcribed 235 times, and Iris's own TTS ("hit a snag, rate
// limit. Want to take a look?", 15:31:52Z) came through at −11 dBFS peak —
// VPIO only cancels audio rendered through ITS OWN output, never another
// process's, yet cost 15 CPU-s/min. Anything he said over it was scored as
// one mixed segment and dropped by the speaker gate.
//
// Tap mode (default when "System Audio Recording" is granted): a CoreAudio
// process tap of ALL output (every app, Iris's speech, our own cues) is the
// far-end reference for a SpeexDSP MDF adaptive canceller + residual echo
// suppressor (capture/aec_core.c), on the plain microphone. The canceller
// does no work while nothing plays (silent reference). Without the
// permission (or on any tap failure) the helper keeps the VPIO path.
// Force a path: --vp / --tap, or ALWAYS_CAPTURE_AEC=vp|tap.

typealias TCCPreflightFn = @convention(c) (CFString, CFDictionary?) -> Int
typealias TCCRequestFn = @convention(c) (CFString, CFDictionary?, @escaping (Bool) -> Void) -> Void
let tccLib = dlopen("/System/Library/PrivateFrameworks/TCC.framework/Versions/A/TCC", RTLD_NOW)
/// "authorized" | "denied" | "undetermined" | "unknown" for System Audio Recording.
func tapPermission() -> String {
    guard let tccLib, let sym = dlsym(tccLib, "TCCAccessPreflight") else { return "unknown" }
    let r = unsafeBitCast(sym, to: TCCPreflightFn.self)("kTCCServiceAudioCapture" as CFString, nil)
    return r == 0 ? "authorized" : r == 1 ? "denied" : "undetermined"
}
/// Shows the system prompt (attributed to the responsible app, Always).
func requestTapPermission() -> Bool {
    guard let tccLib, let sym = dlsym(tccLib, "TCCAccessRequest") else { return false }
    let sem = DispatchSemaphore(value: 0)
    var ok = false
    unsafeBitCast(sym, to: TCCRequestFn.self)("kTCCServiceAudioCapture" as CFString, nil) { g in ok = g; sem.signal() }
    _ = sem.wait(timeout: .now() + 120)
    return ok
}

let hostTicksPerSec: Double = {
    var tb = mach_timebase_info_data_t()
    mach_timebase_info(&tb)
    return 1e9 * Double(tb.denom) / Double(tb.numer)
}()

/// 16 kHz reference samples indexed by host time, so the mic frame captured
/// at host time T reads what was being played at T (the echo of it arrives
/// later — output latency + air — which the adaptive filter models). Writer:
/// the tap IOProc. Reader: the mic tap thread.
final class RefRing {
    private var lock = os_unfair_lock()
    private let cap = 1 << 16  // 4.1 s
    private var buf: [Float]
    /// Side channel 0.5·(L−R) on the same timeline (mono = 0.5·(L+R) in `buf`): stereo
    /// music's left/right difference is invisible to a mono canceller.
    private var side: [Float]
    private var base: UInt64 = 0      // host time of index 0
    private var writeIdx: Int64 = 0   // next index to write
    init() { buf = [Float](repeating: 0, count: 1 << 16); side = [Float](repeating: 0, count: 1 << 16) }
    // Diagnostics (10 s stats line): timestamp deviation of tap writes from
    // the continuous write position, and how often it forced a re-sync.
    var maxDev: Int64 = 0, devSum: Double = 0, writes = 0, resyncs = 0, readsAhead = 0
    func takeStats() -> (Int64, Double, Int, Int) {
        os_unfair_lock_lock(&lock); defer { os_unfair_lock_unlock(&lock) }
        let r = (maxDev, writes > 0 ? devSum / Double(writes) : 0, resyncs, readsAhead)
        maxDev = 0; devSum = 0; writes = 0; resyncs = 0; readsAhead = 0
        return r
    }
    private func index(_ host: UInt64) -> Int64 {
        Int64((Double(Int64(bitPattern: host &- base)) / hostTicksPerSec * 16_000).rounded())
    }
    func write(_ p: UnsafePointer<Float>, _ sd: UnsafePointer<Float>?, _ n: Int, host: UInt64) {
        os_unfair_lock_lock(&lock); defer { os_unfair_lock_unlock(&lock) }
        if base == 0 { base = host }
        var idx = index(host)
        // Stay continuous through timestamp jitter (< 2 ms); re-sync otherwise.
        let dev = idx - writeIdx
        if writeIdx != 0 { maxDev = max(maxDev, abs(dev)); devSum += Double(abs(dev)); writes += 1 }
        if abs(dev) < 32 { idx = writeIdx } else if writeIdx != 0 { resyncs += 1 }
        if idx > writeIdx {  // gap: what played then is unknown → silence
            var j = max(writeIdx, idx - Int64(cap))
            while j < idx { buf[Int(j & Int64(cap - 1))] = 0; side[Int(j & Int64(cap - 1))] = 0; j += 1 }
        }
        for i in 0..<n { buf[Int((idx + Int64(i)) & Int64(cap - 1))] = p[i]; side[Int((idx + Int64(i)) & Int64(cap - 1))] = sd?[i] ?? 0 }
        writeIdx = idx + Int64(n)
    }
    /// The side channel for the same span `read` returned (call right after it, same host time).
    func readSide(_ out: UnsafeMutablePointer<Float>, _ n: Int, host: UInt64) {
        os_unfair_lock_lock(&lock); defer { os_unfair_lock_unlock(&lock) }
        let idx = base == 0 ? Int64.min / 2 : index(host)
        for i in 0..<n {
            let j = idx + Int64(i)
            out[i] = (j < writeIdx && j >= writeIdx - Int64(cap)) ? side[Int(j & Int64(cap - 1))] : 0
        }
    }
    func read(_ out: UnsafeMutablePointer<Float>, _ n: Int, host: UInt64) {
        os_unfair_lock_lock(&lock); defer { os_unfair_lock_unlock(&lock) }
        let idx = base == 0 ? Int64.min / 2 : index(host)
        if base != 0 && idx + Int64(n) > writeIdx { readsAhead += 1 }
        for i in 0..<n {
            let j = idx + Int64(i)
            out[i] = (j < writeIdx && j >= writeIdx - Int64(cap)) ? buf[Int(j & Int64(cap - 1))] : 0
        }
    }
}
let refRing = RefRing()

let tapDriftCompensation = ProcessInfo.processInfo.environment["ALWAYS_CAPTURE_TAP_DRIFT"] == "1"

/// Global tap of every process's output (nothing muted, nothing rerouted) on
/// a private aggregate device; its IOProc downmixes to mono, resamples to
/// 16 kHz and fills `refRing`.
@available(macOS 14.2, *)
final class OutputTap {
    var tapID = AudioObjectID(kAudioObjectUnknown)
    var aggID = AudioObjectID(kAudioObjectUnknown)
    var procID: AudioDeviceIOProcID?
    var resampler: OpaquePointer?
    var mono = [Float](repeating: 0, count: 8192)
    var res = [Float](repeating: 0, count: 8192)
    var sideIn = [Float](repeating: 0, count: 8192)
    var sideRes = [Float](repeating: 0, count: 8192)
    var sideResampler: OpaquePointer?
    var stereo = false
    var rate: Double = 0
    var rateChecked = false, rateT0: UInt64 = 0, rateFrames = 0
    /// Digital silence on the output (nothing playing): input frames of it in a
    /// row, and the fractional 16 kHz sample carried between skipped callbacks.
    var silentFrames = 0
    var silentFrac: Double = 0
    let zeros = [Float](repeating: 0, count: 8192)
    /// Silent input frames that flush the resamplers' filter memory before
    /// resampling is skipped (quality 4 at 192→16 kHz is 768 taps; generous).
    static let silenceFlushFrames = 4096

    func start() -> String? {
        var outDev = AudioObjectID(kAudioObjectUnknown)
        var a = caAddr(kAudioHardwarePropertyDefaultOutputDevice)
        var sz = UInt32(MemoryLayout<AudioObjectID>.size)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &a, 0, nil, &sz, &outDev) == noErr
        else { return "no default output" }
        var uidRef: Unmanaged<CFString>?
        a = caAddr(kAudioDevicePropertyDeviceUID)
        sz = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard AudioObjectGetPropertyData(outDev, &a, 0, nil, &sz, &uidRef) == noErr, let outUID = uidRef?.takeRetainedValue()
        else { return "no output uid" }
        // Iris's duck-tap replays every other app's audio while she speaks
        // (it mutes the originals): tapping it too would put the media into
        // the reference twice. Excluded by name when running now; the
        // watcher below respawns the helper once if it appears later.
        let excluded = audioProcesses().filter { obj in caPid(obj).map { procName($0) == "duck-tap" } ?? false }
        if !excluded.isEmpty { log("ref tap excludes duck-tap (\(excluded.count) process object(s))") }
        let d = CATapDescription(stereoGlobalTapButExcludeProcesses: excluded)
        d.uuid = UUID()
        d.name = "always-capture-ref"
        d.isPrivate = true
        d.muteBehavior = .unmuted
        var st = AudioHardwareCreateProcessTap(d, &tapID)
        if st != noErr { return "AudioHardwareCreateProcessTap \(st)" }
        let desc: [String: Any] = [
            kAudioAggregateDeviceNameKey: "always-capture-ref",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceMainSubDeviceKey: outUID as String,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outUID as String]],
            // The tap and the aggregate's main sub-device are the SAME output
            // device (one clock): drift compensation only adds a hunting
            // varispeed to the reference. Live 2026-10-16 the reference fed to
            // the canceller had mic coherence 0.01 above 300 Hz (vs 0.76 with a
            // sample-exact reference on the same speakers/mic/level) and ERLE
            // 1.5-2 dB — the signature of ~±1 ms jitter (reproduced offline).
            // ALWAYS_CAPTURE_TAP_DRIFT=1 restores it.
            kAudioAggregateDeviceTapListKey: [[kAudioSubTapDriftCompensationKey: tapDriftCompensation,
                                               kAudioSubTapUIDKey: d.uuid.uuidString]],
        ]
        st = AudioHardwareCreateAggregateDevice(desc as CFDictionary, &aggID)
        if st != noErr { stop(); return "AudioHardwareCreateAggregateDevice \(st)" }
        // Tap stream format (the IOProc's input).
        var asbd = AudioStreamBasicDescription()
        a = caAddr(kAudioTapPropertyFormat)
        sz = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        st = AudioObjectGetPropertyData(tapID, &a, 0, nil, &sz, &asbd)
        if st != noErr || asbd.mSampleRate <= 0 { stop(); return "tap format \(st)" }
        rate = asbd.mSampleRate
        // The IOProc runs at the AGGREGATE device's rate, which follows its
        // main sub-device (the speakers), not the tap's nominal format. Live
        // 2026-10-16: tap 48 kHz, speakers at 96 kHz; resampling 96 kHz data
        // as 48 kHz wrote 170 reference samples per 85 samples of time, the
        // ring re-synced on every callback (dev 85 samples, 15k re-syncs per
        // 10 s) and the canceller's ERLE was 0.9-2 dB.
        var aggRate: Float64 = 0
        a = caAddr(kAudioDevicePropertyNominalSampleRate)
        sz = UInt32(MemoryLayout<Float64>.size)
        if AudioObjectGetPropertyData(aggID, &a, 0, nil, &sz, &aggRate) == noErr, aggRate > 0, aggRate != rate {
            log("ref tap: tap format \(Int(rate)) Hz, aggregate runs at \(Int(aggRate)) Hz — resampling from \(Int(aggRate))")
            rate = aggRate
        }
        var err: Int32 = 0
        resampler = speex_resampler_init(1, UInt32(rate), 16_000, 4, &err)
        sideResampler = speex_resampler_init(1, UInt32(rate), 16_000, 4, &err)
        if resampler == nil { stop(); return "resampler \(err)" }
        st = AudioDeviceCreateIOProcIDWithBlock(&procID, aggID, nil) { [unowned self] _, inData, inTime, _, _ in
            self.ioProc(inData, inTime)
        }
        if st != noErr { stop(); return "AudioDeviceCreateIOProcIDWithBlock \(st)" }
        st = AudioDeviceStart(aggID, procID)
        if st != noErr { stop(); return "AudioDeviceStart \(st)" }
        log("ref tap on: global output tap \(Int(rate)) Hz \(asbd.mChannelsPerFrame)ch via \(outUID)")
        return nil
    }

    private func ioProc(_ inData: UnsafePointer<AudioBufferList>, _ inTime: UnsafePointer<AudioTimeStamp>) {
        let ins = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inData))
        guard let b = ins.first, let p = b.mData else { return }
        let f = p.assumingMemoryBound(to: Float.self)
        var n = 0
        if b.mNumberChannels >= 2 {
            let ch = Int(b.mNumberChannels)
            n = min(Int(b.mDataByteSize) / 4 / ch, mono.count)
            for i in 0..<n { mono[i] = 0.5 * (f[i * ch] + f[i * ch + 1]); sideIn[i] = 0.5 * (f[i * ch] - f[i * ch + 1]) }
            stereo = true
        } else if ins.count >= 2, let p2 = ins[1].mData {
            let g = p2.assumingMemoryBound(to: Float.self)
            n = min(Int(b.mDataByteSize) / 4, mono.count)
            for i in 0..<n { mono[i] = 0.5 * (f[i] + g[i]); sideIn[i] = 0.5 * (f[i] - g[i]) }
            stereo = true
        } else {
            n = min(Int(b.mDataByteSize) / 4, mono.count)
            for i in 0..<n { mono[i] = f[i] }
            stereo = false
        }
        guard n > 0 else { return }
        // Verify the data rate from the callbacks themselves (frames per host
        // second over the first ~2 s); a nominal rate that disagrees by > 3%
        // is replaced by the nearest standard rate, once, and logged.
        let hostNow = (inTime.pointee.mFlags.contains(.hostTimeValid)) ? inTime.pointee.mHostTime : mach_absolute_time()
        if !rateChecked {
            if rateT0 == 0 { rateT0 = hostNow; rateFrames = 0 } else {
                let secs = Double(hostNow &- rateT0) / hostTicksPerSec
                if secs >= 2 {
                    let measured = Double(rateFrames) / secs
                    rateChecked = true
                    if abs(measured / rate - 1) > 0.03 {
                        let std = [44_100.0, 48_000, 88_200, 96_000, 176_400, 192_000].min { abs($0 - measured) < abs($1 - measured) }!
                        log(String(format: "ref tap: measured %.0f frames/s vs %.0f assumed — resampling from %.0f", measured, rate, std))
                        rate = std
                        speex_resampler_set_rate(resampler, UInt32(std), 16_000)
                    speex_resampler_set_rate(sideResampler, UInt32(std), 16_000)
                    } else {
                        log(String(format: "ref tap: measured %.0f frames/s (assumed %.0f) ok", measured, rate))
                    }
                }
            }
            rateFrames += n
        }
        // Nothing playing: the tap still delivers every buffer (all zeros) at
        // the output rate, and resampling zeros through two sinc resamplers
        // was this helper's whole idle cost (CPU sweep 2026-10-17: ~2% of a
        // core, `resampler_basic_direct_single`). Once the filters have been
        // flushed by real zeros, write the zeros the resamplers would have
        // produced (same count over time, same host-time position) without
        // running them. Exact zero only — any sound at all takes the full path.
        var peak: Float = 0
        mono.withUnsafeBufferPointer { vDSP_maxmgv($0.baseAddress!, 1, &peak, vDSP_Length(n)) }
        if peak == 0 && stereo {
            sideIn.withUnsafeBufferPointer { vDSP_maxmgv($0.baseAddress!, 1, &peak, vDSP_Length(n)) }
        }
        if peak == 0 {
            let flushed = silentFrames >= OutputTap.silenceFlushFrames
            silentFrames = min(silentFrames + n, Int.max / 2)
            if flushed {
                silentFrac += Double(n) * 16_000 / rate
                let out = min(Int(silentFrac), zeros.count)
                silentFrac -= Double(out)
                let host = (inTime.pointee.mFlags.contains(.hostTimeValid)) ? inTime.pointee.mHostTime : mach_absolute_time()
                if out > 0 { zeros.withUnsafeBufferPointer { refRing.write($0.baseAddress!, nil, out, host: host) } }
                return
            }
        } else {
            silentFrames = 0
            silentFrac = 0
        }
        var inLen = UInt32(n), outLen = UInt32(res.count)
        mono.withUnsafeBufferPointer { mp in
            res.withUnsafeMutableBufferPointer { rp in
                _ = speex_resampler_process_float(resampler, 0, mp.baseAddress, &inLen, rp.baseAddress, &outLen)
            }
        }
        let host = (inTime.pointee.mFlags.contains(.hostTimeValid)) ? inTime.pointee.mHostTime : mach_absolute_time()
        var sIn = UInt32(n), sOut = UInt32(sideRes.count)
        if stereo {
            sideIn.withUnsafeBufferPointer { sp in sideRes.withUnsafeMutableBufferPointer { op in
                _ = speex_resampler_process_float(sideResampler, 0, sp.baseAddress, &sIn, op.baseAddress, &sOut) } }
        }
        res.withUnsafeBufferPointer { r in sideRes.withUnsafeBufferPointer { sd in
            refRing.write(r.baseAddress!, stereo && sOut == outLen ? sd.baseAddress! : nil, Int(outLen), host: host) } }
    }

    func stop() {
        if let procID { AudioDeviceStop(aggID, procID); AudioDeviceDestroyIOProcID(aggID, procID) }
        procID = nil
        if aggID != kAudioObjectUnknown { AudioHardwareDestroyAggregateDevice(aggID) }
        if tapID != kAudioObjectUnknown { AudioHardwareDestroyProcessTap(tapID) }
        aggID = AudioObjectID(kAudioObjectUnknown); tapID = AudioObjectID(kAudioObjectUnknown)
        if let resampler { speex_resampler_destroy(resampler) }
        resampler = nil
        if let sideResampler { speex_resampler_destroy(sideResampler) }
        sideResampler = nil
    }
}

/// Application Support/Always (ALWAYS_CAPTURE_SUPPORT_DIR overrides it for tests).
func alwaysSupportDir() -> URL {
    if let d = ProcessInfo.processInfo.environment["ALWAYS_CAPTURE_SUPPORT_DIR"] { return URL(fileURLWithPath: d) }
    return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/Always")
}
func extractorWeightsPath() -> String {
    alwaysSupportDir().appendingPathComponent("tse/current.bin").path
}
func extractorMtime() -> Date? {
    (try? FileManager.default.attributesOfItem(atPath: extractorWeightsPath()))?[.modificationDate] as? Date
}
/// Echo recordings for the extractor's training (OFF unless he opts in by
/// creating Application Support/Always/tse/dump-echo): EVERY 10 ms frame —
/// the raw 16 kHz mic, the reference it got (aligned exactly as the
/// canceller saw them), what this helper wrote to the daemon (`.out`), and
/// one byte per frame (`.ran`: 1 = canceller ran) — go to
/// tse/echo/<start>.{mic,ref,out}.s16 + .ran (0600), at most
/// `ECHO_DUMP_MAX_SECS` per dump. Local only. (Until 2026-10-16 only
/// frames where the canceller ran were kept: a quiet-room sentence was
/// half missing and a session could not be replayed faithfully.)
let ECHO_DUMP_MAX_SECS = 900.0
final class EchoDump {
    /// mic, ref, side, out, ran, idx — created on the first flush, so a helper
    /// that never captures (a respawn onto a silent mic) leaves no empty dump.
    private var files: [FileHandle]? = nil
    private let stem: String
    var frames = 0
    /// Frames are kept here and written every `FLUSH_SAMPLES` (0.5 s): six
    /// write(2) calls per half second instead of five per 10 ms frame plus a
    /// `Data` per stream per frame. Plain write(2), not `FileHandle.write`,
    /// which raises (abort) on a full disk.
    private static let FLUSH_SAMPLES = 8_000
    private var bMic = [Int16](), bRef = [Int16](), bSide = [Int16](), bOut = [Int16](), bRan = [UInt8](), bIdx = [UInt8]()
    init?() {
        let sup = alwaysSupportDir().appendingPathComponent("tse")
        guard FileManager.default.fileExists(atPath: sup.appendingPathComponent("dump-echo").path) else { return nil }
        let dir = sup.appendingPathComponent("echo")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        stem = dir.appendingPathComponent("\(Int(Date().timeIntervalSince1970))").path
        bMic.reserveCapacity(EchoDump.FLUSH_SAMPLES + 320); bRef.reserveCapacity(EchoDump.FLUSH_SAMPLES + 320)
        bSide.reserveCapacity(EchoDump.FLUSH_SAMPLES + 320); bOut.reserveCapacity(EchoDump.FLUSH_SAMPLES + 320)
        log("echo dump on: \(stem).{mic,ref,out}.s16 + .ran, every frame, mic pre-gained (max \(Int(ECHO_DUMP_MAX_SECS)) s)")
    }
    private func open() -> [FileHandle]? {
        if let files { return files }
        var hs: [FileHandle] = []
        for name in ["mic.s16", "ref.s16", "side.s16", "out.s16", "ran", "idx"] {
            FileManager.default.createFile(atPath: "\(stem).\(name)", contents: nil, attributes: [.posixPermissions: 0o600])
            guard let h = FileHandle(forWritingAtPath: "\(stem).\(name)") else { return nil }
            hs.append(h)
        }
        files = hs
        return hs
    }
    deinit { flush() }
    /// The mic is written pre-gained exactly as the canceller sees it (no int16 clipping).
    func write(_ m: UnsafePointer<Float>, _ r: UnsafePointer<Float>, _ sd: UnsafePointer<Float>, _ n: Int, ran didRun: Bool, micGain: Float) {
        guard Double(frames * n) / 16_000 < ECHO_DUMP_MAX_SECS else { return }
        // Wall clock of every 50th frame (0.5 s): maps the dump onto the logs.
        if frames % 50 == 0 {
            bIdx.append(contentsOf: Array("\(Int(Date().timeIntervalSince1970 * 1000)) \(frames * n)\n".utf8))
        }
        frames += 1
        for i in 0..<n {
            bMic.append(Int16(max(-32768, min(32767, m[i] * micGain * 32768))))
            bRef.append(Int16(max(-32768, min(32767, r[i] * 32768))))
            bSide.append(Int16(max(-32768, min(32767, sd[i] * 32768))))
        }
        bRan.append(didRun ? 1 : 0)
    }
    /// The frame exactly as written to the daemon (after canceller, extractor, AGC).
    /// One `.out` frame per dumped `.mic` frame: after the cap `frames` stops
    /// growing, and the old time check let `.out` run on past it (the
    /// 2026-10-18 dump: out 32.4 MB against mic 28.8 MB).
    private var outFrames = 0
    func writeOut(_ p: UnsafePointer<Int16>, _ n: Int) {
        guard outFrames < frames else { return }
        outFrames += 1
        bOut.append(contentsOf: UnsafeBufferPointer(start: p, count: n))
        if bOut.count >= EchoDump.FLUSH_SAMPLES || Double(frames * n) / 16_000 >= ECHO_DUMP_MAX_SECS { flush() }
    }
    private func flush() {
        func put(_ h: FileHandle, _ p: UnsafeRawBufferPointer) {
            var o = 0
            while o < p.count { let k = Darwin.write(h.fileDescriptor, p.baseAddress! + o, p.count - o); if k <= 0 { return }; o += k }
        }
        if !(bMic.isEmpty && bOut.isEmpty), let f = open() {
            for (h, b) in [(f[0], bMic), (f[1], bRef), (f[2], bSide), (f[3], bOut)] where !b.isEmpty { b.withUnsafeBytes { put(h, $0) } }
            if !bRan.isEmpty { bRan.withUnsafeBytes { put(f[4], $0) } }
            if !bIdx.isEmpty { bIdx.withUnsafeBytes { put(f[5], $0) } }
        }
        bMic.removeAll(keepingCapacity: true); bRef.removeAll(keepingCapacity: true); bSide.removeAll(keepingCapacity: true)
        bOut.removeAll(keepingCapacity: true); bRan.removeAll(keepingCapacity: true); bIdx.removeAll(keepingCapacity: true)
    }
}
// Checked every 10 s (stats timer): the flag appearing starts a dump, the
// flag removed stops it — no helper restart needed.
var echoDumpLock = os_unfair_lock()
var echoDumpShared: EchoDump? = EchoDump()
func echoDumpFlagPresent() -> Bool {
    FileManager.default.fileExists(atPath: alwaysSupportDir().appendingPathComponent("tse/dump-echo").path)
}
func echoDumpRefresh() {
    let want = echoDumpFlagPresent()
    os_unfair_lock_lock(&echoDumpLock)
    let have = echoDumpShared != nil
    os_unfair_lock_unlock(&echoDumpLock)
    if want && !have {
        let d = EchoDump()
        os_unfair_lock_lock(&echoDumpLock); echoDumpShared = d; os_unfair_lock_unlock(&echoDumpLock)
    } else if !want && have {
        os_unfair_lock_lock(&echoDumpLock); echoDumpShared = nil; os_unfair_lock_unlock(&echoDumpLock)
        log("echo dump off (flag removed)")
    }
}

/// Loads the extractor (nil when disabled, or weights/voiceprint missing).
func loadExtractor() -> OpaquePointer? {
    if ProcessInfo.processInfo.environment["ALWAYS_CAPTURE_TSE"] == "0" { log("tse off (ALWAYS_CAPTURE_TSE=0)"); return nil }
    let sup = alwaysSupportDir()
    let weights = sup.appendingPathComponent("tse/current.bin").path
    guard FileManager.default.fileExists(atPath: weights) else { log("tse off (no \(weights))"); return nil }
    guard let d = try? Data(contentsOf: sup.appendingPathComponent("voiceprint.json")),
          let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
          let vp = (j["voiceprint"] as? [NSNumber])?.map({ $0.floatValue }), vp.count == 256
    else { log("tse off (no voiceprint)"); return nil }
    let t = vp.withUnsafeBufferPointer { tse_new(weights, $0.baseAddress!, 256) }
    log(t == nil ? "tse off (weights rejected: \(weights))" : "tse on: \(weights)")
    return t
}

/// Mic → 10 ms frames → canceller (against the reference at the same host
/// time) → speech AGC → s16 on stdout. Tap thread only.
final class TapPipeline {
    let aec: OpaquePointer
    let agc: OpaquePointer
    var acc = [Float](repeating: 0, count: 8192)
    var accN = 0
    var accHost: UInt64 = 0          // host time of acc[0]
    var ref = [Float](repeating: 0, count: Int(AEC_FRAME))
    var side = [Float](repeating: 0, count: Int(AEC_FRAME))
    var out = [Float](repeating: 0, count: Int(AEC_FRAME))
    var pcm = [Int16](repeating: 0, count: Int(AEC_FRAME))
    var engaged = false
    // Stats for the 10 s summary log line.
    var framesRun = 0, framesPassed = 0, maxRefRms: Float = 0
    // Raw mic peak (float, before any int16 conversion) and how many samples
    // were at/over full scale: tells whether loud playback clips IN the
    // driver (peak stays <= 0 dBFS) or only in our int16 canceller (> 0 dBFS).
    var micPeak: Float = 0, micOver = 0
    // Echo reduction actually achieved (canceller in vs out energy while it runs).
    var erleIn: Double = 0, erleOut: Double = 0
    /// Linear pre-gain applied to the mic before the canceller (and to the dump's mic).
    var micGain: Float = 1
    let stereoAec = true
    let echoSuppress: Int32 = -40, echoSuppressActive: Int32 = -1
    var aecUs: Double = 0, aecFrames = 0
    var refL = [Float](repeating: 0, count: Int(AEC_FRAME)), refR = [Float](repeating: 0, count: Int(AEC_FRAME))
    init() {
        let env = ProcessInfo.processInfo.environment
        let tail = Int32(env["ALWAYS_CAPTURE_AEC_TAIL"] ?? "") ?? 4096
        // Canceller on the raw mic, then a speech AGC on its output starting
        // at 22 dB — the measured raw-vs-VPIO level offset — so the daemon's
        // levels do not drop when the path changes.
        // −12 dB before the int16 canceller: at full playback volume the raw
        // float mic peaks at +8.5 dBFS (probe 2026-10-16, 1.4% of samples
        // over full scale) and the int16 conversion hard-clipped it — the
        // driver does not. The AGC below starts 12 dB higher to compensate.
        let pre: Float = -12
        micGain = powf(10, pre / 20)
        // Two-speaker canceller (left/right reference).
        // Residual echo suppressor depth (speex echo_suppress / echo_suppress_active, negative dB; a positive
        // value turns the suppressor off). Speex blends the two by its speech probability, so "active" is what
        // HIS speech gets. Measured 2026-10-17 on the calibration recording with his accepted speech mixed in
        // (12 clips, near -14.7 / far -22.9 dBFS, volumes 50/75/100, music + film + Iris singly and together):
        // speex default -40/-15 cost his far voice 2.3-11.9 dB; -40/-1 costs 1.2-7.8 dB while still removing
        // 13.7 dB of echo on average (default 16.6, off 7.2); mean gate score near 0.423 -> 0.435, far 0.279 -> 0.298;
        // echo-only windows never above 0.26 (gate 0.5) for any setting. -> -40/-1 (`echoSuppress*`).
        aec = aec_new_mc(tail, echoSuppress, echoSuppressActive, pre, stereoAec ? 2 : 1)
        let g0 = Float(env["ALWAYS_CAPTURE_GAIN_DB"] ?? "") ?? 22
        agc = aec_agc_new(-26, 35 - pre, max(0, g0 - pre))
        tse = loadExtractor()
    }
    // His personal extractor (capture/tse_core.c) on the canceller output,
    // only while the reference plays. Weights: Application Support/Always/
    // tse/current.bin, conditioned on his voiceprint.json. Missing either,
    // or ALWAYS_CAPTURE_TSE=0: the canceller output passes unchanged.
    var tse: OpaquePointer?
    // Hot reload: a newly promoted model (current.bin mtime changed) is
    // loaded off the tap thread and swapped in at the next frame.
    var tseLock = os_unfair_lock()
    var tsePending: OpaquePointer?
    var tseHasPending = false
    var tseMtime: Date? = extractorMtime()
    func checkExtractorReload() {
        let m = extractorMtime()
        guard m != tseMtime else { return }
        tseMtime = m
        let t = loadExtractor()
        os_unfair_lock_lock(&tseLock); tsePending = t; tseHasPending = true; os_unfair_lock_unlock(&tseLock)
    }
    var tseOut = [Float](repeating: 0, count: Int(AEC_FRAME))
    var tseActive = false
    var tseFrames = 0, tseUs: Double = 0
    func feed(_ p: UnsafePointer<Float>, _ n: Int, host: UInt64) {
        let frame = Int(AEC_FRAME)
        // The buffer's first sample is at `host`; anything still accumulated
        // is the 16 kHz audio just before it.
        accHost = host &- UInt64(Double(accN) / 16_000 * hostTicksPerSec)
        var i = 0
        while i < n {
            let k = min(n - i, acc.count - accN)
            for j in 0..<k { acc[accN + j] = p[i + j] }
            accN += k; i += k
            var off = 0
            while accN - off >= frame {
                let fh = accHost &+ UInt64(Double(off) / 16_000 * hostTicksPerSec)
                ref.withUnsafeMutableBufferPointer { refRing.read($0.baseAddress!, frame, host: fh) }
                var e: Float = 0
                for r in ref { e += r * r }
                maxRefRms = max(maxRefRms, sqrtf(e / Float(frame)))
                for j in 0..<frame { let v = abs(acc[off + j]); if v > micPeak { micPeak = v }; if v >= 0.9999 { micOver += 1 } }
                side.withUnsafeMutableBufferPointer { refRing.readSide($0.baseAddress!, frame, host: fh) }
                let ran: Int32
                let aecT0 = mach_absolute_time()
                defer { aecUs += Double(mach_absolute_time() - aecT0) / hostTicksPerSec * 1e6; aecFrames += 1 }
                if stereoAec {
                    for j in 0..<frame { refL[j] = ref[j] + side[j]; refR[j] = ref[j] - side[j] }
                    ran = acc.withUnsafeBufferPointer { a in refL.withUnsafeBufferPointer { l in refR.withUnsafeBufferPointer { r in
                        out.withUnsafeMutableBufferPointer { o in aec_process_st(aec, a.baseAddress! + off, l.baseAddress!, r.baseAddress!, o.baseAddress!) } } } }
                } else {
                    ran = acc.withUnsafeBufferPointer { a in
                        ref.withUnsafeBufferPointer { r in
                            out.withUnsafeMutableBufferPointer { o in
                                aec_process(aec, a.baseAddress! + off, r.baseAddress!, o.baseAddress!)
                            }
                        }
                    }
                }
                if ran != 0 {
                    framesRun += 1
                    var ei: Float = 0, eo: Float = 0
                    for j in 0..<frame { ei += acc[off + j] * acc[off + j]; eo += out[j] * out[j] }
                    eo /= micGain * micGain  // canceller output is pre-gained; ERLE compares like with like
                    erleIn += Double(ei); erleOut += Double(eo)
                } else { framesPassed += 1 }
                os_unfair_lock_lock(&echoDumpLock); let echoDump = echoDumpShared; os_unfair_lock_unlock(&echoDumpLock)
                if let d = echoDump {
                    acc.withUnsafeBufferPointer { a in ref.withUnsafeBufferPointer { r in side.withUnsafeBufferPointer { sd in
                        d.write(a.baseAddress! + off, r.baseAddress!, sd.baseAddress!, frame, ran: ran != 0, micGain: micGain) } } }
                }
                if (ran != 0) != engaged {
                    engaged = ran != 0
                    log("aec \(engaged ? "engaged" : "bypassed") (tap reference \(engaged ? "playing" : "silent"))")
                }
                if tseHasPending {
                    os_unfair_lock_lock(&tseLock)
                    let old = tse; tse = tsePending; tsePending = nil; tseHasPending = false; tseActive = false
                    os_unfair_lock_unlock(&tseLock)
                    if let old { tse_free(old) }
                }
                if let tse {
                    if ran != 0 {
                        let t0 = mach_absolute_time()
                        ref.withUnsafeBufferPointer { r in
                            out.withUnsafeBufferPointer { o in
                                tseOut.withUnsafeMutableBufferPointer { tse_process(tse, o.baseAddress!, r.baseAddress!, $0.baseAddress!) }
                            }
                        }
                        for j in 0..<frame { out[j] = tseOut[j] }
                        tseUs += Double(mach_absolute_time() - t0) / hostTicksPerSec * 1e6
                        tseFrames += 1
                        tseActive = true
                    } else if tseActive {
                        tse_reset(tse)
                        tseActive = false
                    }
                }
                // Gain adapts only while nothing plays: music never trains it.
                out.withUnsafeMutableBufferPointer { aec_agc_process(agc, $0.baseAddress!, ran != 0 ? 0 : 1) }
                for j in 0..<frame {
                    let s = out[j] * 32768
                    pcm[j] = Int16(max(-32768, min(32767, s)))
                }
                if let d = echoDump { pcm.withUnsafeBufferPointer { d.writeOut($0.baseAddress!, frame) } }
                pcm.withUnsafeBytes { writeAll($0.baseAddress!, frame * 2) }
                off += frame
            }
            if off > 0 {
                for j in 0..<(accN - off) { acc[j] = acc[off + j] }
                accN -= off
                accHost = accHost &+ UInt64(Double(off) / 16_000 * hostTicksPerSec)
            }
        }
    }
}

var tapStatsHolder: DispatchSourceTimer?
var signalSources: [DispatchSourceSignal] = []
var grantWatch: DispatchSourceTimer?
let aecPref = ProcessInfo.processInfo.environment["ALWAYS_CAPTURE_AEC"] ?? ""
if args.contains("--request-tap-permission") {
    log("tap permission before: \(tapPermission())")
    log("tap permission granted: \(requestTapPermission())")
    exit(0)
}
if args.contains("--tap-status") { log("tap permission: \(tapPermission())"); exit(0) }
var tapMode = false
var outputTap: AnyObject?
// Diagnostics: `--ref-only` runs only the output tap (no microphone) and
// writes the 16 kHz mono s16 reference to stdout, 10 ms at a time.
if args.contains("--ref-only") {
    guard #available(macOS 14.2, *) else { exit(5) }
    let t = OutputTap()
    if let err = t.start() { log("ref tap unavailable: \(err)"); exit(5) }
    outputTap = t
    atexit { if #available(macOS 14.2, *), let t = outputTap as? OutputTap { t.stop() } }
    for sig in [SIGTERM, SIGINT, SIGHUP] { signal(sig, { _ in exit(0) }) }
    var f = [Float](repeating: 0, count: 160), pcm = [Int16](repeating: 0, count: 160)
    while true {
        usleep(10_000)
        f.withUnsafeMutableBufferPointer { refRing.read($0.baseAddress!, 160, host: mach_absolute_time() - UInt64(0.2 * hostTicksPerSec)) }
        for i in 0..<160 { pcm[i] = Int16(max(-32768, min(32767, f[i] * 32768))) }
        pcm.withUnsafeBytes { p in var o = 0; while o < 320 { let n = write(1, p.baseAddress! + o, 320 - o); if n <= 0 { exit(0) }; o += n } }
    }
}
if !args.contains("--no-vp") && !args.contains("--vp") && aecPref != "vp" {
    if #available(macOS 14.2, *) {
        let perm = tapPermission()
        if perm == "authorized" || args.contains("--tap") || aecPref == "tap" {
            let t = OutputTap()
            if let err = t.start() { log("ref tap unavailable (\(err), permission=\(perm)) — plain mic, no echo cancelling") }
            else {
                outputTap = t; tapMode = true
                // duck-tap (Iris) started after this tap: respawn once so the
                // new tap excludes it (the daemon restarts the helper).
                let excludedAtStart = audioProcesses().contains { obj in caPid(obj).map { procName($0) == "duck-tap" } ?? false }
                if !excludedAtStart {
                    let w = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
                    w.schedule(deadline: .now() + 10, repeating: 10)
                    w.setEventHandler {
                        if audioProcesses().contains(where: { obj in caPid(obj).map { procName($0) == "duck-tap" } ?? false }) {
                            log("duck-tap appeared — respawning so the reference tap excludes it"); exit(0)
                        }
                    }
                    w.resume()
                    grantWatch = w
                }
            }
        } else {
            log("ref tap needs System Audio Recording (permission=\(perm)) — plain mic, no echo cancelling")
            // The Always app is the sole requester (Settings / onboarding
            // "System Audio Recording"). Here: watch for the grant (a cheap
            // TCC preflight every 10 s) and exit once it lands, so the
            // daemon respawns this helper straight into tap mode.
            do {
                let w = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
                w.schedule(deadline: .now() + 10, repeating: 10)
                w.setEventHandler {
                    if tapPermission() == "authorized" { log("System Audio Recording granted — respawning into tap mode"); exit(0) }
                }
                w.resume()
                grantWatch = w
            }
        }
    }
}
// VPIO only on explicit request (`--vp` / ALWAYS_CAPTURE_AEC=vp). Running
// it changes what EVERYONE hears: macOS puts the output through voice
// processing and ducks other apps even at the minimum ducking level — with
// Always quit his music was "fuller and louder" (2026-10-15). Without the
// System Audio Recording grant the mic is captured plain; the "My Voice"
// gate already rejects other voices.
let useVP = (args.contains("--vp") || aecPref == "vp") && !args.contains("--no-vp") && !tapMode

// Tap teardown on EVERY exit path (EPIPE, parent death, device change,
// SIGTERM/SIGINT): destroy the aggregate device and the process tap
// explicitly rather than leaving it to coreaudiod's client cleanup. If
// coreaudiod is wedged and the teardown blocks, a 2 s watchdog _exits anyway
// (2026-10-15 19:27Z: a hung coreaudiod left a tap helper unkillable).
var tapTornDown = false
func tearDownTap() {
    guard !tapTornDown else { return }
    tapTornDown = true
    Thread { sleep(2); _exit(0) }.start()
    if #available(macOS 14.2, *), let t = outputTap as? OutputTap { t.stop() }
}
atexit { tearDownTap() }
for sig in [SIGTERM, SIGINT, SIGHUP] {
    signal(sig, SIG_IGN)
    let src = DispatchSource.makeSignalSource(signal: sig, queue: .global())
    src.setEventHandler { exit(0) }
    src.resume()
    signalSources.append(src)
}
let tapPipeline: TapPipeline? = tapMode ? TapPipeline() : nil

let engine = AVAudioEngine()
let input = engine.inputNode

if useVP {
    do {
        try input.setVoiceProcessingEnabled(true)
    } catch {
        log("voice processing unavailable: \(error) — exiting so the daemon falls back to rec")
        exit(4)
    }
    input.isVoiceProcessingAGCEnabled = args.contains("--agc")
    if #available(macOS 14.0, *) {
        input.voiceProcessingOtherAudioDuckingConfiguration =
            AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
                enableAdvancedDucking: false,
                duckingLevel: args.contains("--duck") ? .default : .min)
    }
}

let inFormat = input.outputFormat(forBus: 0)
log("input format: \(inFormat) vp=\(useVP) presentation_latency_ms=\(Int(input.presentationLatency * 1000))")
guard inFormat.sampleRate > 0, inFormat.channelCount > 0 else {
    log("no input device")
    exit(5)
}
// VPIO on macOS can expose several channels (the processed signal is channel
// 0); mix down only channel 0 rather than averaging in raw mic channels.
let monoIn = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: inFormat.sampleRate,
                           channels: 1, interleaved: false)!
// Tap mode converts to float for the canceller, which writes the s16.
let outFormat = AVAudioFormat(commonFormat: tapMode ? .pcmFormatFloat32 : .pcmFormatInt16, sampleRate: 16_000,
                              channels: 1, interleaved: true)!
guard let converter = AVAudioConverter(from: monoIn, to: outFormat) else {
    log("no converter")
    exit(6)
}

let stdoutFd = FileHandle.standardOutput.fileDescriptor
func writeAll(_ ptr: UnsafeRawPointer, _ count: Int) {
    var off = 0
    while off < count {
        let n = write(stdoutFd, ptr + off, count - off)
        if n <= 0 {
            if n < 0 && errno == EINTR { continue }
            exit(0)  // daemon closed the pipe
        }
        off += n
    }
}

// ---- Double-talk evidence: processed vs raw mic levels ----------------------
// 2026-10-14 16:36–16:39Z the user's own speech under music came out of the
// voice processor near-silent (peak −71 dBFS, STT heard nothing) or scored
// −0.10…0.25 against his voiceprint, and passed again the moment the music
// paused. Whether VPIO suppressed him (double-talk) cannot be told from the
// processed stream alone, so while the canceller is engaged a second, plain
// (non-voice-processing) input engine meters the RAW microphone and both
// levels are logged once per second of audible activity. Metering only —
// the raw signal is never sent to the daemon. Measured: starting/stopping the
// raw engine raises no configuration change on the VPIO engine; with VPIO
// running the raw device level reads ~20 dB below the processed one.
// Disable: ALWAYS_CAPTURE_RAW_LEVELS=0.
final class LevelMeter {
    private var lock = os_unfair_lock()
    private var sumSq = 0.0, n = 0, peak: Float = 0, voicedMs = 0.0
    let voicedDbfs: Float
    init(voicedDbfs: Float) { self.voicedDbfs = voicedDbfs }
    /// Accumulate one channel of samples at `rate` Hz (10 ms frames).
    func add(_ p: UnsafePointer<Float>, _ count: Int, rate: Double) {
        let frame = max(1, Int(rate / 100))
        let thr = powf(10, voicedDbfs / 20)
        var s = 0.0, pk: Float = 0, voiced = 0
        var i = 0
        while i < count {
            let end = min(count, i + frame)
            var fs: Float = 0
            for j in i..<end { let v = p[j]; fs += v * v; pk = max(pk, abs(v)) }
            s += Double(fs)
            if sqrtf(fs / Float(end - i)) >= thr { voiced += end - i }
            i = end
        }
        os_unfair_lock_lock(&lock)
        sumSq += s; n += count; peak = max(peak, pk); voicedMs += Double(voiced) * 1000 / rate
        os_unfair_lock_unlock(&lock)
    }
    /// (rms dBFS, peak dBFS, voiced ms) since the last call; resets.
    func take() -> (Float, Float, Int) {
        os_unfair_lock_lock(&lock)
        defer { sumSq = 0; n = 0; peak = 0; voicedMs = 0; os_unfair_lock_unlock(&lock) }
        let rms = n > 0 ? Float(10 * log10(max(sumSq / Double(n), 1e-12))) : -120
        return (rms, 20 * log10f(max(peak, 1e-6)), Int(voicedMs))
    }
}
let procMeter = LevelMeter(voicedDbfs: -55)
let rawMeter = LevelMeter(voicedDbfs: -75)
let rawLevelsEnabled = useVP && ProcessInfo.processInfo.environment["ALWAYS_CAPTURE_RAW_LEVELS"] != "0"
var rawEngine: AVAudioEngine?   // main queue only

func startRawMeter() {
    guard rawLevelsEnabled, rawEngine == nil else { return }
    let e = AVAudioEngine()
    let f = e.inputNode.outputFormat(forBus: 0)
    guard f.sampleRate > 0, f.channelCount > 0 else { return }
    e.inputNode.installTap(onBus: 0, bufferSize: 2048, format: f) { b, _ in
        guard let ch = b.floatChannelData?[0] else { return }
        rawMeter.add(ch, Int(b.frameLength), rate: f.sampleRate)
    }
    do { e.prepare(); try e.start(); rawEngine = e; _ = rawMeter.take() }
    catch { log("raw meter unavailable: \(error)") }
}

func stopRawMeter() {
    rawEngine?.stop()
    rawEngine = nil
}

// Tap buffers are reused across callbacks (the tap fires ~40×/s; allocating
// two PCM buffers per callback was pure malloc churn). Only the tap thread
// touches them.
var monoBuf: AVAudioPCMBuffer?
var outBuf: AVAudioPCMBuffer?
/// The mic's blocks leave CoreAudio's IO thread through an AVAudioSinkNode
/// into a lock-free ring (no allocation, no lock, no wait on that thread)
/// and are processed on one user-interactive thread. Before 2026-10-19 they
/// came through `installTap`, whose delivery queue fell behind while the
/// display slept: "mic tap gap 568 ms (buffer captured 571 ms ago)" at
/// 02:16:18Z, and 5.6-12.7 s bursts at 00:09-00:41Z reached the daemon as one
/// late backlog. A gap past 500 ms between processed blocks is still logged.
let micRing = rt_ring_create(1 << 19)!   // floats: ~5.4 s at 96 kHz
let micReady = DispatchSemaphore(value: 0)
let micStride = inFormat.isInterleaved ? UInt32(inFormat.channelCount) : 1
/// The processing thread is woken once per ~2048 input frames (~21 ms at
/// 96 kHz, the tap's cadence), not per IO cycle.
let MIC_WAKE_FRAMES: UInt32 = 2048
var micPendingFrames: UInt32 = 0   // IO thread only
let micSink = AVAudioSinkNode { ts, frameCount, abl -> OSStatus in
    let list = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: abl))
    guard frameCount > 0, let first = list.first, let data = first.mData else { return noErr }
    let host = ts.pointee.mFlags.contains(.hostTimeValid) ? ts.pointee.mHostTime : mach_absolute_time()
    _ = rt_ring_write(micRing, data.assumingMemoryBound(to: Float.self), frameCount, micStride, host)
    micPendingFrames += frameCount
    if micPendingFrames >= MIC_WAKE_FRAMES {
        micPendingFrames = 0
        micReady.signal()
    }
    return noErr
}
engine.attach(micSink)
engine.connect(input, to: micSink, format: inFormat)

var lastTapAt: UInt64 = 0
func processMic(_ src: UnsafePointer<Float>, _ frames: Int, host: UInt64) {
    let tapNow = mach_absolute_time()
    if lastTapAt != 0 {
        let gapMs = Double(tapNow &- lastTapAt) / hostTicksPerSec * 1000
        if gapMs >= 500 {
            let lateMs = Double(Int64(bitPattern: tapNow &- host)) / hostTicksPerSec * 1000
            log(String(format: "mic tap gap %.0f ms (buffer captured %.0f ms ago)", gapMs, lateMs))
        }
    }
    lastTapAt = tapNow
    guard frames > 0 else { return }
    if rawLevelsEnabled { procMeter.add(src, frames, rate: inFormat.sampleRate) }
    if monoBuf == nil || Int(monoBuf!.frameCapacity) < frames {
        monoBuf = AVAudioPCMBuffer(pcmFormat: monoIn, frameCapacity: AVAudioFrameCount(max(frames, 4096)))
    }
    guard let mono = monoBuf else { return }
    mono.frameLength = AVAudioFrameCount(frames)
    mono.floatChannelData![0].update(from: src, count: frames)
    let cap = AVAudioFrameCount(Double(frames) * 16_000 / inFormat.sampleRate) + 32
    if outBuf == nil || outBuf!.frameCapacity < cap {
        outBuf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: max(cap, 2048))
    }
    guard let out = outBuf else { return }
    out.frameLength = 0
    var fed = false
    var err: NSError?
    converter.convert(to: out, error: &err) { _, status in
        if fed { status.pointee = .noDataNow; return nil }
        fed = true
        status.pointee = .haveData
        return mono
    }
    if let err { log("convert error: \(err)"); return }
    let n = Int(out.frameLength)
    guard n > 0 else { return }
    if let tp = tapPipeline {
        tp.feed(out.floatChannelData![0], n, host: host)
    } else {
        writeAll(out.int16ChannelData![0], n * 2)
    }
}

/// Everything queued since the last wake is converted and processed as ONE
/// block (stamped with its first IO block's capture time): the converter and
/// canceller run ~47 times a second as with the tap, not once per IO cycle.
let micThread = Thread {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
    let cap = 1 << 16
    var block = [Float](repeating: 0, count: cap)
    var host: UInt64 = 0, firstHost: UInt64 = 0
    while true {
        micReady.wait()
        var filled = 0
        while filled < cap - 8192 {
            let n = block.withUnsafeMutableBufferPointer {
                rt_ring_read(micRing, $0.baseAddress! + filled, UInt32(cap - filled), &host)
            }
            if n == 0 { break }
            if filled == 0 { firstHost = host }
            filled += Int(n)
        }
        if filled > 0 { block.withUnsafeBufferPointer { processMic($0.baseAddress!, filled, host: firstHost) } }
    }
}
micThread.name = "mic-process"
micThread.qualityOfService = .userInteractive
micThread.start()

// ---- Status cues through THIS engine's output -------------------------------
// Audio rendered by the voice-processing unit's own output is never ducked by
// it (it is the "far end" a voice app must hear) and is the echo-cancellation
// reference, so cues neither get ducked nor leak into the mic. The daemon
// sends `play <path>` lines on stdin (2026-10-14: cues played from the daemon
// process were inaudible while this unit ran).
let cueFormat = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 2)!
let cuePlayers: [AVAudioPlayerNode] = (0..<6).map { _ in AVAudioPlayerNode() }
for p in cuePlayers {
    engine.attach(p)
    engine.connect(p, to: engine.mainMixerNode, format: cueFormat)
}
var cueCache: [String: AVAudioPCMBuffer] = [:]
var nextCuePlayer = 0
let cueQueue = DispatchQueue(label: "always-capture.cues")

func loadCue(_ path: String) -> AVAudioPCMBuffer? {
    if let b = cueCache[path] { return b }
    guard let file = try? AVAudioFile(forReading: URL(fileURLWithPath: path)),
          let src = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)),
          (try? file.read(into: src)) != nil else { return nil }
    var buf = src
    if src.format != cueFormat {
        guard let conv = AVAudioConverter(from: src.format, to: cueFormat),
              let dst = AVAudioPCMBuffer(pcmFormat: cueFormat,
                  frameCapacity: AVAudioFrameCount(Double(src.frameLength) * 48_000 / src.format.sampleRate) + 64)
        else { return nil }
        var fed = false
        var err: NSError?
        conv.convert(to: dst, error: &err) { _, st in
            if fed { st.pointee = .endOfStream; return nil }
            fed = true; st.pointee = .haveData; return src
        }
        if err != nil { return nil }
        buf = dst
    }
    cueCache[path] = buf
    return buf
}

func playCue(_ path: String) {
    let started = DispatchTime.now().uptimeNanoseconds
    guard let buf = loadCue(path) else { log("cue_failed load \(path)"); return }
    let p = cuePlayers[nextCuePlayer % cuePlayers.count]
    nextCuePlayer += 1
    // A cue rendered while the canceller is bypassed would leak into the mic:
    // engage it for the cue's duration (+ echo tail) first.
    let cueSecs = Double(buf.frameLength) / cueFormat.sampleRate
    DispatchQueue.main.sync { aecHoldFor(seconds: cueSecs + 0.5) }
    p.stop()
    p.scheduleBuffer(buf, at: nil, options: [], completionHandler: nil)
    p.play()
    let ms = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6
    log(String(format: "cue_played %@ playing=%d ms=%.1f", (path as NSString).lastPathComponent, p.isPlaying ? 1 : 0, ms))
}

/// Dry run (launch smoke test): decode + render the cue OFFLINE through a
/// player node into a buffer — no output device, nothing audible — and log
/// the rendered frame count and peak, proving the file plays.
func dryRenderCue(_ path: String) {
    guard let buf = loadCue(path) else { log("cue_failed load \(path)"); return }
    let off = AVAudioEngine()
    let node = AVAudioPlayerNode()
    off.attach(node)
    off.connect(node, to: off.mainMixerNode, format: cueFormat)
    do {
        try off.enableManualRenderingMode(.offline, format: cueFormat, maximumFrameCount: 4096)
        try off.start()
    } catch { log("cue_failed dry \(error)"); return }
    node.scheduleBuffer(buf, at: nil, options: [], completionHandler: nil)
    node.play()
    let out = AVAudioPCMBuffer(pcmFormat: off.manualRenderingFormat, frameCapacity: 4096)!
    var frames = 0
    var peak: Float = 0
    while frames < Int(buf.frameLength) {
        guard (try? off.renderOffline(4096, to: out)) == .success else { break }
        let n = Int(out.frameLength)
        if let ch = out.floatChannelData?[0] { for i in 0..<n { peak = max(peak, abs(ch[i])) } }
        frames += n
    }
    off.stop()
    log(String(format: "cue_rendered %@ frames=%d peak=%.4f", (path as NSString).lastPathComponent, frames, peak))
}

Thread {
    while let line = readLine(strippingNewline: true) {
        let parts = line.split(separator: " ", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { continue }
        switch parts[0] {
        case "play": cueQueue.async { playCue(parts[1]) }
        case "load": cueQueue.async { _ = loadCue(parts[1]) }
        case "dry": cueQueue.async { dryRenderCue(parts[1]) }
        default: break
        }
    }
    // stdin closed: the daemon is gone or stopped sending cues; capture
    // continues until stdout closes.
}.start()

func applyDucking() {
    guard useVP else { return }
    if #available(macOS 14.0, *) {
        input.voiceProcessingOtherAudioDuckingConfiguration =
            AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
                enableAdvancedDucking: false,
                duckingLevel: args.contains("--duck") ? .default : .min)
        let c = input.voiceProcessingOtherAudioDuckingConfiguration
        log("ducking advanced=\(c.enableAdvancedDucking) level=\(c.duckingLevel.rawValue)")
    }
}

// ---- Echo canceller only while something is playing -------------------------
// All state below is touched on the main queue only (CoreAudio listener blocks
// are delivered there; cue holds hop there).
let aecAlways = args.contains("--aec-always")
let selfPid = getpid()
var aecEngaged = true                 // VPIO starts un-bypassed
var aecHoldUntil = DispatchTime.now() // cue hold
var aecOffWork: DispatchWorkItem?
var aecWatched = Set<AudioObjectID>()
/// Keep the canceller engaged this long after the last output stops: apps
/// stop/start their output between tracks, and the room still rings briefly.
let aecHangover = 2.0
/// After MEDIA output stops (a video or music app, not our own TTS/cues),
/// stay engaged much longer: paused media resumes, and the canceller can
/// only engage in REACTION to output starting — so a resume after bypass
/// put the first seconds of the video into the mic uncancelled, still
/// converging. Measured 2026-10-14: a YouTube resume at 11:25:14Z ~71 s
/// after its pause put the narrator's first 1.5 s through as the user
/// (0.581) and it was pasted, while every narrator window under a converged
/// canceller scored 0.15–0.21. That day's WebKit pause→resume gaps: 30 s,
/// 88 s, 71 s — all inside 120 s. Cost: VPIO stays on (≈15 vs 3 CPU-s/min)
/// for up to 2 min after media pauses. Override:
/// ALWAYS_CAPTURE_MEDIA_HANGOVER (seconds).
let aecMediaHangover = Double(ProcessInfo.processInfo.environment["ALWAYS_CAPTURE_MEDIA_HANGOVER"] ?? "") ?? 120.0
/// Short-lived players whose output ends for good when they exit: the
/// ordinary hangover suffices (Iris's speech player; system sound players).
let aecShortOutputNames: Set<String> = ["pcm-play", "afplay", "always-daemon"]
var aecLastOutputWasMedia = false

func caAddr(_ sel: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: sel, mScope: kAudioObjectPropertyScopeGlobal,
                               mElement: kAudioObjectPropertyElementMain)
}

@available(macOS 14.2, *)
func audioProcesses() -> [AudioObjectID] {
    var addr = caAddr(kAudioHardwarePropertyProcessObjectList)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size) == noErr,
          size > 0 else { return [] }
    var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
    guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &ids) == noErr
    else { return [] }
    return Array(ids.prefix(Int(size) / MemoryLayout<AudioObjectID>.size))
}

func caU32(_ obj: AudioObjectID, _ sel: AudioObjectPropertySelector) -> UInt32? {
    var addr = caAddr(sel)
    var v: UInt32 = 0
    var size = UInt32(MemoryLayout<UInt32>.size)
    return AudioObjectGetPropertyData(obj, &addr, 0, nil, &size, &v) == noErr ? v : nil
}

func caPid(_ obj: AudioObjectID) -> pid_t? {
    var addr = caAddr(kAudioProcessPropertyPID)
    var v: pid_t = 0
    var size = UInt32(MemoryLayout<pid_t>.size)
    return AudioObjectGetPropertyData(obj, &addr, 0, nil, &size, &v) == noErr ? v : nil
}

/// System daemons that keep an output stream running permanently without
/// playing anything audible (measured 2026-10-14: `corespeechd` — Siri's
/// voice trigger — reports running input AND output around the clock).
/// Counting them would pin the canceller on forever.
let aecIgnoredNames: Set<String> = Set(["corespeechd"]).union(
    (ProcessInfo.processInfo.environment["ALWAYS_CAPTURE_AEC_IGNORE"] ?? "")
        .split(separator: ",").map(String.init))  // extra names, for side-by-side tests

func procName(_ pid: pid_t) -> String {
    var buf = [CChar](repeating: 0, count: 256)
    return proc_name(pid, &buf, UInt32(buf.count)) > 0 ? String(cString: buf) : ""
}

/// PIDs of OTHER processes currently running audio output.
@available(macOS 14.2, *)
func otherOutputPids() -> [pid_t] {
    audioProcesses().compactMap { obj in
        guard let pid = caPid(obj), pid != selfPid,
              caU32(obj, kAudioProcessPropertyIsRunningOutput) == 1,
              !aecIgnoredNames.contains(procName(pid)) else { return nil }
        return pid
    }
}

func setAecEngaged(_ on: Bool, _ why: String) {
    guard on != aecEngaged else { return }
    aecEngaged = on
    input.isVoiceProcessingBypassed = !on
    if on { startRawMeter() } else { stopRawMeter() }
    // The tap format must not change with the state (the converter and the
    // 16 kHz mono s16 stream are built once); logged so a test can prove it.
    let f = input.outputFormat(forBus: 0)
    log("aec \(on ? "engaged" : "bypassed") (\(why)) tap=\(f.channelCount)ch@\(Int(f.sampleRate))")
}

/// Main queue only.
func aecReevaluate() {
    guard useVP, !aecAlways else { return }
    guard #available(macOS 14.2, *) else { return }
    let playing = otherOutputPids()
    let held = DispatchTime.now() < aecHoldUntil
    if !playing.isEmpty || held {
        if !playing.isEmpty {
            aecLastOutputWasMedia = playing.contains { !aecShortOutputNames.contains(procName($0)) }
        }
        aecOffWork?.cancel(); aecOffWork = nil
        setAecEngaged(true, held && playing.isEmpty ? "cue" : "output pids \(playing.map { "\($0):\(procName($0))" })")
        if held && playing.isEmpty {
            // Re-check when the hold expires. The item must clear itself
            // first: left set, the `aecOffWork == nil` guard below refused
            // to arm the bypass timer, pinning the canceller on after every
            // cue until some other process happened to play (measured
            // 2026-10-14: engaged 08:55:58 → 09:00:23 after one cue).
            let w = DispatchWorkItem { aecOffWork = nil; aecReevaluate() }
            aecOffWork = w
            DispatchQueue.main.asyncAfter(deadline: aecHoldUntil + 0.05, execute: w)
        }
        return
    }
    guard aecEngaged, aecOffWork == nil else { return }
    let hangover = aecLastOutputWasMedia ? aecMediaHangover : aecHangover
    let w = DispatchWorkItem {
        aecOffWork = nil
        if #available(macOS 14.2, *), otherOutputPids().isEmpty, DispatchTime.now() >= aecHoldUntil {
            aecLastOutputWasMedia = false
            setAecEngaged(false, "no other output for \(Int(hangover))s")
        }
    }
    aecOffWork = w
    DispatchQueue.main.asyncAfter(deadline: .now() + hangover, execute: w)
}

/// Main queue only.
func aecHoldFor(seconds: Double) {
    guard useVP, !aecAlways else { return }
    let until = DispatchTime.now() + seconds
    if until > aecHoldUntil { aecHoldUntil = until }
    aecReevaluate()
}

/// Listen for processes appearing/disappearing and for each one's
/// "running output" flag. Main queue only.
@available(macOS 14.2, *)
func aecWatchProcesses() {
    let procs = audioProcesses()
    aecWatched.formIntersection(procs)
    for obj in procs where !aecWatched.contains(obj) {
        // IsRunningOutput is READ to decide, but its change is NOT
        // notified (measured 2026-10-14, macOS 27: an afplay object
        // registers with out=0, flips to out=1 ~0.6 s later, and a
        // listener on 'piro' never fires — only 'pir?' (IsRunning) and
        // 'pdv#' (Devices) do). Engagement therefore waited for some
        // unrelated process-list change: a YouTube resume was engaged
        // 4.8 s late (11:25:10→14.9Z) and its narrator pasted as the user.
        var ok = false
        for sel in [kAudioProcessPropertyIsRunningOutput, kAudioProcessPropertyIsRunning,
                    kAudioProcessPropertyDevices] {
            var addr = caAddr(sel)
            if AudioObjectAddPropertyListenerBlock(obj, &addr, .main, { _, _ in aecReevaluate() }) == noErr {
                ok = true
            }
        }
        if ok { aecWatched.insert(obj) }
    }
}

func startAecGate() {
    guard useVP else { return }
    if aecAlways { log("aec always engaged (--aec-always)"); return }
    guard #available(macOS 14.2, *) else { log("aec gate unavailable (< macOS 14.2): always engaged"); return }
    var addr = caAddr(kAudioHardwarePropertyProcessObjectList)
    AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &addr, .main) { _, _ in
        aecWatchProcesses()
        aecReevaluate()
    }
    aecWatchProcesses()
    log("aec gate on: other output pids now \(otherOutputPids().map { "\($0):\(procName($0))" }) (bypassed after \(aecHangover)s of none)")
    aecReevaluate()
}

do {
    engine.prepare()
    try engine.start()
} catch {
    log("engine start failed: \(error)")
    exit(7)
}
// Re-applied once running: the ducking configuration is only guaranteed to
// take effect on a running voice-processing unit.
applyDucking()
startAecGate()
if aecEngaged { startRawMeter() }
// Kept at top level: a timer source released at the end of a scope never fires.
let levelTimer = DispatchSource.makeTimerSource(queue: .main)
if rawLevelsEnabled {
    // Once per second while the canceller is engaged, only when something
    // was audible in either stream (a quiet room logs nothing).
    levelTimer.schedule(deadline: .now() + 1, repeating: 1)
    levelTimer.setEventHandler {
        let (pr, pp, pv) = procMeter.take()
        let (rr, rp, rv) = rawMeter.take()
        guard aecEngaged, rawEngine != nil, pv > 0 || rv > 0 else { return }
        log(String(format: "levels media=%d vp=%.1f/%.1f voiced=%d raw=%.1f/%.1f voiced=%d",
                   aecLastOutputWasMedia ? 1 : 0, pr, pp, pv, rr, rp, rv))
    }
    levelTimer.resume()
}
if let tp = tapPipeline {
    // One line per 10 s while anything played: canceller duty cycle + gain.
    let tapStatsTimer = DispatchSource.makeTimerSource(queue: .main)
    tapStatsTimer.schedule(deadline: .now() + 10, repeating: 10)
    tapStatsTimer.setEventHandler {
        tp.checkExtractorReload()
        echoDumpRefresh()
        // Torn reads of these counters are harmless (diagnostics only).
        let run = tp.framesRun, passed = tp.framesPassed, peak = tp.maxRefRms
        let mpk = tp.micPeak, mover = tp.micOver
        let aus = tp.aecFrames > 0 ? tp.aecUs / Double(tp.aecFrames) : 0; tp.aecUs = 0; tp.aecFrames = 0
        tp.framesRun = 0; tp.framesPassed = 0; tp.maxRefRms = 0; tp.micPeak = 0; tp.micOver = 0
        if run > 0 {
            let tf = tp.tseFrames, tu = tp.tseUs
            tp.tseFrames = 0; tp.tseUs = 0
            let erle = tp.erleOut > 0 ? 10 * log10(tp.erleIn / tp.erleOut) : 0
            let micDb = 10 * log10(max(tp.erleIn / Double(max(run, 1) * 160), 1e-12))
            tp.erleIn = 0; tp.erleOut = 0
            let (md, ad, rs, ra) = refRing.takeStats()
            log(String(format: "ref ring: write dev max %d avg %.1f samples, resyncs %d, mic frames reading past the ref %d", md, ad, rs, ra))
            log(String(format: "aec stats run=%d passed=%d ref_max=%.1f dBFS gain=%.1f dB erle=%.1f dB mic=%.1f dBFS tse=%d frames %.0f us/frame drift_comp=%d mic_peak=%.1f dBFS mic_full_scale=%d frame_us=%.0f stereo=%d es=%d/%d pregain=%.0f dB", run, passed,
                       20 * log10f(max(peak, 1e-6)), aec_agc_gain_db(tp.agc), erle, micDb, tf, tf > 0 ? tu / Double(tf) : 0, tapDriftCompensation ? 1 : 0,
                       20 * log10f(max(mpk, 1e-6)), mover, aus, tp.stereoAec ? 1 : 0, tp.echoSuppress, tp.echoSuppressActive, 20 * log10f(tp.micGain)))
        }
    }
    tapStatsTimer.resume()
    tapStatsHolder = tapStatsTimer
}
log("capturing mode=\(tapMode ? "tap-aec" : useVP ? "vpio" : "plain")")

// Parent-death watchdog: the daemon may be SIGKILLed with our stdout pipe
// still referenced elsewhere; never outlive it. Kernel exit notification
// (kqueue NOTE_EXIT) instead of a 1 Hz getppid() poll.
let parentWatch = DispatchSource.makeProcessSource(identifier: parentPid, eventMask: .exit, queue: .main)
parentWatch.setEventHandler { exit(0) }
parentWatch.resume()
if getppid() != parentPid { exit(0) }  // parent died before the source armed
NotificationCenter.default.addObserver(
    forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil
) { _ in
    // Device change / route change: exit so the daemon respawns us bound to
    // the new default input (same as its RespawnRecorder path for rec).
    log("engine configuration changed — exiting for respawn")
    exit(0)
}
RunLoop.main.run()
