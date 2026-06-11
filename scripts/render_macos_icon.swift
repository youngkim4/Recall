import AppKit
import Foundation

// Recall app icon: the setting sun. A vermilion disc dissolving into
// paper stripes as it sinks -- a moment passing into memory. Warm
// geometry on cream paper.

let outputPath = CommandLine.arguments.dropFirst().first ?? "/tmp/RecallIcon-1024.png"
let outputURL = URL(fileURLWithPath: outputPath)
let canvas: CGFloat = 1024
let image = NSImage(size: NSSize(width: canvas, height: canvas))

// palette: mirrors app/src/App.css tokens
let paper = NSColor(calibratedRed: 0.957, green: 0.933, blue: 0.886, alpha: 1) // #f4eee2
let paperLight = NSColor(calibratedRed: 0.984, green: 0.965, blue: 0.925, alpha: 1) // #fbf6ec
let ink = NSColor(calibratedRed: 0.165, green: 0.125, blue: 0.090, alpha: 1) // #2a2017
let inkLine = NSColor(calibratedRed: 0.169, green: 0.129, blue: 0.094, alpha: 0.30)
let inkLineSoft = NSColor(calibratedRed: 0.169, green: 0.129, blue: 0.094, alpha: 0.16)
let vermilion = NSColor(calibratedRed: 0.757, green: 0.235, blue: 0.153, alpha: 1) // #c13c27

image.lockFocus()
NSColor.clear.setFill()
NSRect(x: 0, y: 0, width: canvas, height: canvas).fill()

// Big Sur icon grid: 824pt squircle centered on a 1024 canvas
let tileRect = NSRect(x: 100, y: 100, width: 824, height: 824)
let tileRadius: CGFloat = 186
let tile = NSBezierPath(roundedRect: tileRect, xRadius: tileRadius, yRadius: tileRadius)

// soft drop shadow so the paper sits off the dock background
let shadow = NSShadow()
shadow.shadowColor = NSColor.black.withAlphaComponent(0.30)
shadow.shadowOffset = NSSize(width: 0, height: -10)
shadow.shadowBlurRadius = 22
NSGraphicsContext.saveGraphicsState()
shadow.set()
paper.setFill()
tile.fill()
NSGraphicsContext.restoreGraphicsState()

// faint top-light gradient keeps the paper from reading flat-white
let gradient = NSGradient(starting: paperLight, ending: paper)
gradient?.draw(in: tile, angle: -90)

// the sun: vermilion disc, lower half sliced into thinning strips by the
// paper -- geometry doing the remembering
NSGraphicsContext.saveGraphicsState()
tile.addClip()

let sunCenter = NSPoint(x: 512, y: 560)
let sunRadius: CGFloat = 312

let sun = NSBezierPath(ovalIn: NSRect(
    x: sunCenter.x - sunRadius,
    y: sunCenter.y - sunRadius,
    width: sunRadius * 2,
    height: sunRadius * 2
))
vermilion.setFill()
sun.fill()

// cream gaps widen as the sun sinks; each gap re-draws the tile gradient
// clipped to (sun ∩ slice) so the paper stays seamless around the disc
let gaps: [(y: CGFloat, height: CGFloat)] = [
    (488, 24),
    (404, 36),
    (312, 50),
    (204, 66),
]
for gap in gaps {
    NSGraphicsContext.saveGraphicsState()
    sun.addClip()
    NSBezierPath(rect: NSRect(x: 0, y: gap.y, width: canvas, height: gap.height)).addClip()
    gradient?.draw(in: tile, angle: -90)
    NSGraphicsContext.restoreGraphicsState()
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
