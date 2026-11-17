// Make typedstream fixtures with Apple's own encoder.
//
// Messages stores message text in `message.attributedBody` as an
// NSAttributedString archived by NSArchiver (the "streamtyped" format).
// The archive decoder is tested against blobs made here, never against
// blobs made by our own writer.
//
// Print every fixture as "name<TAB>hex":
//   swift scripts/make_typedstream_fixtures.swift
// Write one .hex file per fixture plus manifest.json:
//   swift scripts/make_typedstream_fixtures.swift tests/archive/fixtures/typedstream
//
// All text is invented. No real message is used.

import CryptoKit
import Foundation

struct Fixture {
    let name: String
    let text: String
    let mutable: Bool
    let attributes: String  // none | part | runs | attachment
    let recipe: String      // literal | counter:<bytes> | counter:<bytes>+<suffix>
}

// "00000|00001|..." cut to an exact byte count. Every offset is distinct,
// so a cut or shifted decode shows up as a different string.
func counterText(_ bytes: Int) -> String {
    var out = ""
    out.reserveCapacity(bytes + 6)
    var index = 0
    while out.utf8.count < bytes {
        out += String(format: "%05d|", index)
        index += 1
    }
    return String(out.prefix(bytes))
}

func counterCase(_ bytes: Int, mutable: Bool = false) -> Fixture {
    Fixture(
        name: (mutable ? "mutable_ascii_" : "ascii_") + String(bytes),
        text: counterText(bytes),
        mutable: mutable,
        attributes: "none",
        recipe: "counter:\(bytes)"
    )
}

// Counter text followed by a multi-byte suffix, exact total byte count.
func suffixCase(_ name: String, total: Int, suffix: String) -> Fixture {
    let head = counterText(total - suffix.utf8.count)
    return Fixture(
        name: name,
        text: head + suffix,
        mutable: false,
        attributes: "none",
        recipe: "counter:\(total - suffix.utf8.count)+suffix"
    )
}

func literal(_ name: String, _ text: String, mutable: Bool = false, attributes: String = "none") -> Fixture {
    Fixture(name: name, text: text, mutable: mutable, attributes: attributes, recipe: "literal")
}

let fixtures: [Fixture] = [
    // length marker boundaries: one byte up to 127, 0x81 + 2 bytes up to
    // 32,767, 0x82 + 4 bytes from 32,768
    counterCase(1),
    counterCase(127),
    counterCase(128),
    counterCase(255),
    counterCase(256),
    counterCase(300),
    counterCase(32767),
    counterCase(32768),
    counterCase(65535),
    counterCase(70000),
    // same boundaries with the mutable classes Messages also writes
    counterCase(127, mutable: true),
    counterCase(128, mutable: true),
    counterCase(300, mutable: true),
    // multi-byte endings on both sides of the one-byte limit
    suffixCase("emoji_end_127", total: 127, suffix: "🎉"),
    suffixCase("emoji_end_128", total: 128, suffix: "🎉"),
    suffixCase("emoji_end_130", total: 130, suffix: "🎉"),
    suffixCase("emoji_end_300", total: 300, suffix: "👩‍👩‍👧‍👦"),
    suffixCase("accented_end_200", total: 200, suffix: "é"),
    suffixCase("cjk_end_256", total: 256, suffix: "語"),
    // short literal texts
    literal("empty", ""),
    literal("emoji_end", "See you tonight 🎉"),
    literal("emoji_only", "🎂"),
    literal("accented", "Café déjà vu, señor. Où est le garçon? Æther, naïve, Ångström."),
    literal("cjk_japanese", "今日は寿司を食べました。明日は映画を見に行きます。"),
    literal("cjk_chinese", "我们明天去看电影,好吗?"),
    literal("korean", "오늘 저녁에 같이 밥 먹을래요? 학교에서 만나요."),
    literal("newline", "first line\nsecond line\n\nfourth line"),
    literal("tab_and_spaces", "  two leading spaces\tand a tab  "),
    // characters that a clean-up step must not drop
    literal("zwj_family", "Family \u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467} at the lake"),
    literal("variation_selector", "Love \u{2764}\u{FE0F} and call \u{260E}\u{FE0F}"),
    literal("invisible_marks", "zero\u{200B}width and word\u{2060}joiner and mark\u{200E}"),
    literal("outer_newlines", "\n line with outer newlines \n"),
    literal("mutable_short", "short mutable text", mutable: true),
    literal("mutable_accented", "Crème brûlée à la carte", mutable: true),
    // attributes the way Messages writes them
    literal("part_attribute", "Dinner at seven?", attributes: "part"),
    literal("part_attribute_mutable", "Dinner at eight then.", mutable: true, attributes: "part"),
    literal("two_runs", "Call Rowan about the tickets", attributes: "runs"),
    literal("attachment_only", "\u{FFFC}", attributes: "attachment"),
    literal("attachment_with_caption", "\u{FFFC}Look at this view", attributes: "attachment"),
]

func attributed(_ fixture: Fixture) -> NSAttributedString {
    let result = NSMutableAttributedString(string: fixture.text)
    let whole = NSRange(location: 0, length: result.length)
    switch fixture.attributes {
    case "part":
        result.addAttribute(NSAttributedString.Key("__kIMMessagePartAttributeName"), value: NSNumber(value: 0), range: whole)
    case "runs":
        result.addAttribute(NSAttributedString.Key("__kIMMessagePartAttributeName"), value: NSNumber(value: 0), range: whole)
        result.addAttribute(
            NSAttributedString.Key("__kIMMentionConfirmedMention"),
            value: "+15550100" as NSString,
            range: NSRange(location: 5, length: 5)
        )
    case "attachment":
        result.addAttribute(
            NSAttributedString.Key("__kIMFileTransferGUIDAttributeName"),
            value: "at_0_00000000-0000-4000-8000-000000000001" as NSString,
            range: NSRange(location: 0, length: 1)
        )
        result.addAttribute(NSAttributedString.Key("__kIMMessagePartAttributeName"), value: NSNumber(value: 0), range: whole)
    default:
        break
    }
    return fixture.mutable ? result : NSAttributedString(attributedString: result)
}

func hex(_ data: Data) -> String {
    data.map { String(format: "%02x", $0) }.joined()
}

func wrapped(_ text: String, width: Int = 96) -> String {
    var lines: [String] = []
    var start = text.startIndex
    while start < text.endIndex {
        let end = text.index(start, offsetBy: width, limitedBy: text.endIndex) ?? text.endIndex
        lines.append(String(text[start..<end]))
        start = end
    }
    return lines.joined(separator: "\n") + "\n"
}

let outputDirectory = CommandLine.arguments.dropFirst().first
var manifest: [[String: Any]] = []

for fixture in fixtures {
    let object = attributed(fixture)
    let data = NSArchiver.archivedData(withRootObject: object)
    let utf8 = Data(fixture.text.utf8)
    let digest = SHA256.hash(data: utf8).map { String(format: "%02x", $0) }.joined()
    var entry: [String: Any] = [
        "name": fixture.name,
        "class": String(describing: type(of: object)),
        "mutable": fixture.mutable,
        "attributes": fixture.attributes,
        "recipe": fixture.recipe,
        "utf8_bytes": utf8.count,
        "utf16_units": fixture.text.utf16.count,
        "sha256": digest,
        "blob_bytes": data.count,
    ]
    if utf8.count <= 400 {
        entry["text"] = fixture.text
    }
    manifest.append(entry)

    if let directory = outputDirectory {
        let url = URL(fileURLWithPath: directory).appendingPathComponent(fixture.name + ".hex")
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try wrapped(hex(data)).write(to: url, atomically: true, encoding: .ascii)
    } else {
        print("\(fixture.name)\t\(hex(data))")
    }
}

if let directory = outputDirectory {
    let info = ProcessInfo.processInfo.operatingSystemVersion
    let payload: [String: Any] = [
        "encoder": "NSArchiver.archivedData(withRootObject:)",
        "macos": "\(info.majorVersion).\(info.minorVersion).\(info.patchVersion)",
        "fixtures": manifest,
    ]
    let json = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    let url = URL(fileURLWithPath: directory).appendingPathComponent("manifest.json")
    try json.write(to: url)
    print("wrote \(fixtures.count) fixtures to \(directory)")
}
