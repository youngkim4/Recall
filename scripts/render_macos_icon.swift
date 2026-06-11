import AppKit
import Foundation

// Recall app icon: a typographic open quote -- the product is the exact
// words people said. Cream paper squircle, double hairline frame (the
// card/report motif), two serif quote marks in ink and vermilion.

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

// editorial double hairline frame, the report/card motif
let outerFrame = NSBezierPath(
    roundedRect: tileRect.insetBy(dx: 64, dy: 64),
    xRadius: tileRadius - 58,
    yRadius: tileRadius - 58
)
inkLine.setStroke()
outerFrame.lineWidth = 7
outerFrame.stroke()

let innerFrame = NSBezierPath(
    roundedRect: tileRect.insetBy(dx: 84, dy: 84),
    xRadius: tileRadius - 76,
    yRadius: tileRadius - 76
)
inkLineSoft.setStroke()
innerFrame.lineWidth = 3.5
innerFrame.stroke()

// the mark: a double open quote built from two single-quote glyphs so the
// trailing mark can carry the vermilion accent
func serifFont(size: CGFloat, weight: NSFont.Weight) -> NSFont {
    let base = NSFont.systemFont(ofSize: size, weight: weight)
    if let descriptor = base.fontDescriptor.withDesign(.serif),
       let font = NSFont(descriptor: descriptor, size: size) {
        return font
    }
    return NSFont(name: "Georgia-Bold", size: size) ?? base
}

let quoteFont = serifFont(size: 940, weight: .heavy)
let mark = NSMutableAttributedString()
mark.append(NSAttributedString(string: "\u{2018}", attributes: [
    .font: quoteFont,
    .foregroundColor: ink,
    .kern: 26,
]))
mark.append(NSAttributedString(string: "\u{2018}", attributes: [
    .font: quoteFont,
    .foregroundColor: vermilion,
]))

// quote glyphs hang from the cap line: the line box is mostly empty space
// below them, so center on the glyph mass, not the box
let markSize = mark.size()
let markOrigin = NSPoint(
    x: (canvas - markSize.width) / 2 + 4,
    y: (canvas - markSize.height) / 2 - 205
)
mark.draw(at: markOrigin)

image.unlockFocus()

guard
    let tiff = image.tiffRepresentation,
    let bitmap = NSBitmapImageRep(data: tiff),
    let png = bitmap.representation(using: .png, properties: [:])
else {
    fatalError("Could not encode icon PNG")
}

try png.write(to: outputURL)
