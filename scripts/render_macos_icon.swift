import AppKit
import Foundation

// Recall app icon: the lamp of the app, lit. The same candle that stands
// beside the reasoning control and the steps of the chat (app/src/lib/lamp.ts),
// at one still moment, drawn in square cells on the dark ink of the
// Broadsheet theme. app/public/favicon.svg draws the same cells.

let outputPath = CommandLine.arguments.dropFirst().first ?? "/tmp/RecallIcon-1024.png"
let outputURL = URL(fileURLWithPath: outputPath)
let canvas: CGFloat = 1024
let image = NSImage(size: NSSize(width: canvas, height: canvas))

func rgb(_ hex: Int, _ alpha: CGFloat = 1) -> NSColor {
    NSColor(
        srgbRed: CGFloat((hex >> 16) & 0xff) / 255,
        green: CGFloat((hex >> 8) & 0xff) / 255,
        blue: CGFloat(hex & 0xff) / 255,
        alpha: alpha
    )
}

let plate = rgb(0x1d1b19)
// e the candle, w its stripe of light, o m h the flame from its edge to its
// heart; a digit is the glow, in the middle colour, that many fortieths strong
let cellColors: [Character: NSColor] = [
    "e": rgb(0xe6d3bd), "w": rgb(0xfaf1e4), "o": rgb(0xb54268), "m": rgb(0xe8784a), "h": rgb(0xffdea9),
]
let glow = 0xe8784a

// the cells, top row first, taken from lib/lamp.ts (level 0.8, at 2.3 s)
let grid: [String] = [
    ".....11111.....",
    "...111111111...",
    ".1112222222111.",
    ".11223oo332211.",
    "11233ommoo33211",
    "1123oommmo43211",
    "1223ommhmo43221",
    "12234omhmo43221",
    "122345mhm543221",
    "112334oho433211",
    ".112333e333211.",
    ".1112eeeeee111.",
    "...11eweeee1...",
    ".....eweeee....",
    ".....eweeee....",
    ".....eweeee....",
    ".....eweeee....",
    ".eeeeeeeeeeeeee",
]

image.lockFocus()
NSColor.clear.setFill()
NSRect(x: 0, y: 0, width: canvas, height: canvas).fill()

// Big Sur icon grid: 824pt squircle centered on a 1024 canvas
let tileRect = NSRect(x: 100, y: 100, width: 824, height: 824)
let tileRadius: CGFloat = 186
let tile = NSBezierPath(roundedRect: tileRect, xRadius: tileRadius, yRadius: tileRadius)

// soft drop shadow so the tile sits off the dock background
let shadow = NSShadow()
shadow.shadowColor = NSColor.black.withAlphaComponent(0.30)
shadow.shadowOffset = NSSize(width: 0, height: -10)
shadow.shadowBlurRadius = 22
NSGraphicsContext.saveGraphicsState()
shadow.set()
plate.setFill()
tile.fill()
NSGraphicsContext.restoreGraphicsState()

// the lamp, as tall as about three quarters of the tile, in the middle of it
NSGraphicsContext.saveGraphicsState()
tile.addClip()
NSGraphicsContext.current?.shouldAntialias = false
let rows = grid.count
let cols = grid.first?.count ?? 0
let cell = (600 / CGFloat(rows)).rounded(.down)
let left = ((canvas - cell * CGFloat(cols)) / 2).rounded()
let top = ((canvas + cell * CGFloat(rows)) / 2 + 10).rounded()
for (y, row) in grid.enumerated() {
    for (x, mark) in row.enumerated() where mark != "." {
        let color = mark.wholeNumberValue.map { rgb(glow, CGFloat($0) / 40) } ?? cellColors[mark] ?? .clear
        color.setFill()
        NSRect(x: left + CGFloat(x) * cell, y: top - CGFloat(y + 1) * cell, width: cell, height: cell).fill()
    }
}
NSGraphicsContext.restoreGraphicsState()

image.unlockFocus()

guard
    let tiff = image.tiffRepresentation,
    let bitmap = NSBitmapImageRep(data: tiff),
    let png = bitmap.representation(using: .png, properties: [:])
else {
    fatalError("Could not encode icon PNG")
}

try png.write(to: outputURL)
