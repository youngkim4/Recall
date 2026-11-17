import CryptoKit
import Foundation

// Fixtures for the attributedBody decoder, made by Apple's own encoder.
// A test that builds its own archive repeats the mistake of the decoder.
// All text here is invented: nothing comes from a Messages database.
//
// run from the repo root:
//   swift -suppress-warnings scripts/make_attributed_body_fixtures.swift
// (NSArchiver is deprecated, so Swift warns without the flag)
// The output has no date and no machine name: a second run gives the same bytes.

let outputDir = CommandLine.arguments.dropFirst().first ?? "tests/fixtures/typedstream"
let inlineTextLimit = 512  // bytes: a longer text is in the manifest as a hash only
let hexLineBytes = 32

// the attribute that Messages puts on each part of a message
let partKey = NSAttributedString.Key("__kIMMessagePartAttributeName")

struct Fixture {
    let name: String
    let note: String
    let root: String
    let value: NSObject
    let text: String?  // nil: the archive holds no string
}

func fail(_ message: String) -> Never {
    fputs("make_attributed_body_fixtures: \(message)\n", stderr)
    exit(1)
}

func hex<Bytes: Sequence>(_ bytes: Bytes) -> String where Bytes.Element == UInt8 {
    bytes.map { String(format: "%02x", $0) }.joined()
}

func hexLines(_ data: Data) -> String {
    let bytes = [UInt8](data)
    let lines = stride(from: 0, to: bytes.count, by: hexLineBytes).map {
        hex(bytes[$0..<min($0 + hexLineBytes, bytes.count)])
    }
    return lines.joined(separator: "\n") + "\n"
}

// text of a known size whose content depends on the position
let alphabet = Array("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
func ascii(_ count: Int) -> String {
    String((0..<count).map { alphabet[$0 % alphabet.count] })
}

func message(_ name: String, _ note: String, _ text: String, mutable: Bool = false) -> Fixture {
    let attributes: [NSAttributedString.Key: Any] = [partKey: NSNumber(value: 0)]
    if mutable {
        return Fixture(
            name: "mutable_\(name)", note: "\(note), in a mutable string", root: "NSMutableAttributedString",
            value: NSMutableAttributedString(string: text, attributes: attributes), text: text)
    }
    return Fixture(
        name: name, note: note, root: "NSAttributedString",
        value: NSAttributedString(string: text, attributes: attributes), text: text)
}

let sizes: [(count: Int, note: String)] = [
    (1, "1 byte: the shortest text"),
    (127, "127 bytes: the longest text with a one-byte length"),
    (128, "128 bytes: the shortest text with a two-byte length"),
    (255, "255 bytes: the old decoder lost the last byte from 128 to 255"),
    (256, "256 bytes: the first length with a high byte"),
    (300, "300 bytes: the old decoder kept 43 bytes of this text"),
    (32767, "32,767 bytes: the longest text with a two-byte length"),
    (32768, "32,768 bytes: the shortest text with a four-byte length"),
    (65535, "65,535 bytes: a four-byte length"),
    (70000, "70,000 bytes: a four-byte length"),
]

let grin = "\u{1F600}"
let family = "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}\u{200D}\u{1F466}"
let rainbowFlag = "\u{1F3F3}\u{FE0F}\u{200D}\u{1F308}"
let zwjText = "Family \(family), flag \(rainbowFlag), heart \u{2764}\u{FE0F}"
let endSpaces = "  two spaces before and one after "
let multibyteUnit = "!h\u{E9}llo w\u{F6}rld \u{1F30D}!"  // 20 bytes

let texts: [Fixture] = sizes.map { message("ascii_\($0.count)", $0.note, ascii($0.count)) } + [
    message("emoji_end", "ends in an emoji", "See you at eight \(grin)"),
    message("emoji_end_130", "130 bytes, ends in an emoji: the old decoder split it", ascii(126) + grin),
    message("zwj_emoji", "emoji joined with U+200D, and U+FE0F selectors", zwjText),
    message(
        "emoji_sequences", "tag characters, a keycap and a skin tone",
        "\u{1F3F4}\u{E0067}\u{E0062}\u{E0073}\u{E0063}\u{E0074}\u{E007F} 1\u{FE0F}\u{20E3} \u{1F44D}\u{1F3FD}"),
    message(
        "accented", "accented letters, composed and decomposed",
        "Caf\u{E9} d\u{E9}j\u{E0} vu, na\u{EF}ve fa\u{E7}ade in Z\u{FC}rich. Decomposed: e\u{301} n\u{303}"),
    message(
        "cjk", "Chinese, Japanese and Korean, with U+3000",
        "\u{4ECA}\u{5929}\u{5929}\u{6C14}\u{5F88}\u{597D}\u{3002}\u{3000}\u{660E}\u{65E5}\u{306F}\u{6674}\u{308C}\u{3067}\u{3059}\u{3002} \u{C548}\u{B155}\u{D558}\u{C138}\u{C694}"),
    message("newline", "newlines and a tab inside the text", "first line\nsecond line\n\n\tfourth line, indented"),
    message("spaces_both_ends", "two spaces before the text and one after it", endSpaces),
    message("newline_ends", "a newline before the text and after it", "\nstarts and ends with a newline\n"),
    message("whitespace_only", "only spaces and a newline", "  \n "),
    message(
        "format_chars", "invisible format characters and unusual spaces",
        "zero\u{200B}width, left\u{200E}mark, \u{5E9}\u{5DC}\u{5D5}\u{5DD}\u{200F} right mark, "
            + "word\u{2060}joiner, soft\u{AD}hyphen, no\u{A0}break, \u{2066}isolate\u{2069}, "
            + "line\u{2028}separator, \u{202B}embedding\u{202C}"),
    message("attachment_char", "only U+FFFC, the place of an attachment", "\u{FFFC}"),
    message(
        "control_chars", "NUL, U+0001, U+001F, tab, newline, carriage return and U+007F",
        "a\u{0}b\u{1}c\u{1F}d\te\nf\rg\u{7F}h"),
    message("multibyte_128", "128 bytes of two-byte letters", String(repeating: "\u{E9}", count: 64)),
    message(
        "multibyte_300", "300 bytes: byte 43 is in the middle of a letter",
        String(repeating: multibyteUnit, count: 15)),
    message("empty", "the empty string", ""),
    message("ascii_1", "1 byte", ascii(1), mutable: true),
    message("ascii_128", "128 bytes", ascii(128), mutable: true),
    message("ascii_300", "300 bytes", ascii(300), mutable: true),
    message("zwj_emoji", "emoji joined with U+200D", zwjText, mutable: true),
    message("spaces_both_ends", "spaces at both ends", endSpaces, mutable: true),
]

let fixtures = texts + [
    Fixture(
        name: "not_text", note: "an archive of two numbers: no string in it", root: "NSArray",
        value: NSArray(array: [NSNumber(value: 1), NSNumber(value: 2.5)]), text: nil)
]

// sizes that the names promise
let promised = ["emoji_end_130": 130, "multibyte_128": 128, "multibyte_300": 300]
    .merging(sizes.map { ("ascii_\($0.count)", $0.count) }) { first, _ in first }
for fixture in fixtures {
    if let want = promised[fixture.name], fixture.text?.utf8.count != want {
        fail("\(fixture.name) is not \(want) bytes")
    }
}
if Set(fixtures.map(\.name)).count != fixtures.count {
    fail("two fixtures have the same name")
}

do {
    let directory = URL(fileURLWithPath: outputDir, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

    var entries: [[String: Any]] = []
    for fixture in fixtures {
        let data = NSArchiver.archivedData(withRootObject: fixture.value)
        var entry: [String: Any] = [
            "name": fixture.name,
            "note": fixture.note,
            "root": fixture.root,
            "archive_bytes": data.count,
            "utf8_bytes": NSNull(),
            "sha256": NSNull(),
            "utf8_hex": NSNull(),
        ]
        if let text = fixture.text {
            // read it back with Apple's decoder: the archive must hold this exact text
            let back = (NSUnarchiver.unarchiveObject(with: data) as? NSAttributedString)?.string
            guard let back, Array(back.utf8) == Array(text.utf8) else {
                fail("\(fixture.name) does not read back as the same text")
            }
            let utf8 = Data(text.utf8)
            entry["utf8_bytes"] = utf8.count
            entry["sha256"] = hex(SHA256.hash(data: utf8))
            if utf8.count <= inlineTextLimit {
                entry["utf8_hex"] = hex(utf8)
            }
        } else if data.range(of: Data("NSString".utf8)) != nil {
            fail("\(fixture.name) must not hold a string")
        }
        try hexLines(data).write(
            to: directory.appendingPathComponent("\(fixture.name).hex"), atomically: true, encoding: .ascii)
        entries.append(entry)
        print("\(fixture.name)\t\(fixture.text?.utf8.count ?? 0) bytes of text\t\(data.count) bytes of archive")
    }

    let manifest: [String: Any] = [
        "encoder": "NSArchiver.archivedData(withRootObject:)",
        "generator": "scripts/make_attributed_body_fixtures.swift",
        "fixtures": entries,
    ]
    let json = try JSONSerialization.data(
        withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    try (json + Data("\n".utf8)).write(to: directory.appendingPathComponent("manifest.json"), options: .atomic)
    print("wrote \(fixtures.count) fixtures and manifest.json to \(outputDir)")
} catch {
    fail("\(error.localizedDescription)")
}
