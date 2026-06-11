import AppKit
import Foundation

// Recall app icon: the wordmark reduced to a monogram. Cream paper squircle,
// double hairline frame (the card/report motif), serif ink "R" with a
// vermilion full stop -- "Recall." as one glyph. Warm-paper memoir brand.

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

// the monogram: serif R in warm ink, vermilion full stop
func serifFont(size: CGFloat, weight: NSFont.Weight) -> NSFont {
    let base = NSFont.systemFont(ofSize: size, weight: weight)
    if let descriptor = base.fontDescriptor.withDesign(.serif),
       let font = NSFont(descriptor: descriptor, size: size) {
        return font
    }
    return NSFont(name: "Georgia-Bold", size: size) ?? base
}

let letter = NSMutableAttributedString()
letter.append(NSAttributedString(string: "R", attributes: [
    .font: serifFont(size: 556, weight: .heavy),
    .foregroundColor: ink,
]))
letter.append(NSAttributedString(string: ".", attributes: [
    .font: serifFont(size: 556, weight: .heavy),
    .foregroundColor: vermilion,
    .kern: 6,
]))

let textSize = letter.size()
let textOrigin = NSPoint(
    x: (canvas - textSize.width) / 2 + 10,
    y: (canvas - textSize.height) / 2 + 6
)
letter.draw(at: textOrigin)

image.unlockFocus()

guard
    let tiff = image.tiffRepresentation,
    let bitmap = NSBitmapImageRep(data: tiff),
    let png = bitmap.representation(using: .png, properties: [:])
else {
    fatalError("Could not encode icon PNG")
}

try png.write(to: outputURL)
