// watchtext — watch TextEdit's focused text view through the Accessibility
// API (20 ms polling) and report when text first appears / last changes.
// usage: watchtext <deadline epoch secs> <stable secs>
// prints one JSON line: {"first": epoch|null, "last": epoch|null, "text": "..."}
import AppKit
import ApplicationServices
import Foundation

let deadline = Double(CommandLine.arguments[1])!
let stable = Double(CommandLine.arguments[2])!
guard let app = NSRunningApplication.runningApplications(withBundleIdentifier: "com.apple.TextEdit").first else {
    print("{\"error\":\"TextEdit not running\"}"); exit(1)
}
let ax = AXUIElementCreateApplication(app.processIdentifier)

func currentText() -> String {
    var focused: CFTypeRef?
    guard AXUIElementCopyAttributeValue(ax, kAXFocusedUIElementAttribute as CFString, &focused) == .success,
          let el = focused else { return "" }
    var value: CFTypeRef?
    guard AXUIElementCopyAttributeValue(el as! AXUIElement, kAXValueAttribute as CFString, &value) == .success
    else { return "" }
    return (value as? String) ?? ""
}

var prev = currentText()
let initial = prev
var first: Double? = nil
var last: Double? = nil
while Date().timeIntervalSince1970 < deadline {
    let now = Date().timeIntervalSince1970
    let cur = currentText()
    if cur != prev {
        if first == nil, !cur.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { first = now }
        last = now
        prev = cur
    }
    if let l = last, now - l > stable { break }
    Thread.sleep(forTimeInterval: 0.02)
}
let obj: [String: Any] = ["first": first.map { $0 as Any } ?? NSNull(),
                          "last": last.map { $0 as Any } ?? NSNull(),
                          "text": prev, "initial": initial]
let data = try JSONSerialization.data(withJSONObject: obj)
print(String(data: data, encoding: .utf8)!)
