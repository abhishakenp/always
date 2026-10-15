// playdev — play an audio file to a NAMED output device (not the default),
// so a loopback device (BlackHole) can act as the test "microphone" while the
// system default output stays elsewhere (echo cancellation never sees it as
// the Mac's own playback). Prints the wall-clock start/end (epoch seconds).
// usage: playdev "<device name>" <file>
import AVFoundation
import CoreAudio
import Foundation

func deviceID(named name: String) -> AudioDeviceID? {
    var addr = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices,
                                          mScope: kAudioObjectPropertyScopeGlobal,
                                          mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size) == noErr else { return nil }
    var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
    guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &ids) == noErr else { return nil }
    for id in ids {
        var naddr = AudioObjectPropertyAddress(mSelector: kAudioObjectPropertyName,
                                               mScope: kAudioObjectPropertyScopeGlobal,
                                               mElement: kAudioObjectPropertyElementMain)
        var cf: Unmanaged<CFString>?
        var s = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        if AudioObjectGetPropertyData(id, &naddr, 0, nil, &s, &cf) == noErr,
           let n = cf?.takeRetainedValue() as String?, n == name {
            // must have output streams
            var saddr = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreams,
                                                   mScope: kAudioDevicePropertyScopeOutput,
                                                   mElement: kAudioObjectPropertyElementMain)
            var ss: UInt32 = 0
            if AudioObjectGetPropertyDataSize(id, &saddr, 0, nil, &ss) == noErr, ss > 0 { return id }
        }
    }
    return nil
}

let args = CommandLine.arguments
guard args.count == 3, var dev = deviceID(named: args[1]) else {
    fputs("usage: playdev <output device name> <file>  (device not found)\n", stderr); exit(2)
}
let file = try AVAudioFile(forReading: URL(fileURLWithPath: args[2]))
let engine = AVAudioEngine()
let out = engine.outputNode
guard let unit = out.audioUnit,
      AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0,
                           &dev, UInt32(MemoryLayout<AudioDeviceID>.size)) == noErr else {
    fputs("cannot bind output device\n", stderr); exit(3)
}
let player = AVAudioPlayerNode()
engine.attach(player)
engine.connect(player, to: engine.mainMixerNode, format: file.processingFormat)
try engine.start()
let done = DispatchSemaphore(value: 0)
player.scheduleFile(file, at: nil) { done.signal() }
let t0 = Date().timeIntervalSince1970
player.play()
done.wait()
// The completion fires when the player has consumed the data, ahead of the
// device actually playing it out: wait for the file's real duration.
let duration = Double(file.length) / file.processingFormat.sampleRate
let left = t0 + duration + 0.1 - Date().timeIntervalSince1970
if left > 0 { Thread.sleep(forTimeInterval: left) }
let t1 = Date().timeIntervalSince1970
print(String(format: "%.3f %.3f", t0, t1))
engine.stop()
