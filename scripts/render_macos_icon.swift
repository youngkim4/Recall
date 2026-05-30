import AppKit
import Foundation

let outputPath = CommandLine.arguments.dropFirst().first ?? "/tmp/RecallIcon-1024.png"
let outputURL = URL(fileURLWithPath: outputPath)
let size: CGFloat = 1024
let image = NSImage(size: NSSize(width: size, height: size))

image.lockFocus()
NSColor.clear.setFill()
NSRect(x: 0, y: 0, width: size, height: size).fill()

let tileRect = NSRect(x: 64, y: 64, width: 896, height: 896)
let tile = NSBezierPath(roundedRect: tileRect, xRadius: 216, yRadius: 216)
NSColor.white.setFill()
tile.fill()
NSColor(calibratedRed: 0.847, green: 0.855, blue: 0.886, alpha: 1).setStroke()
tile.lineWidth = 20
tile.stroke()

let letterAttributes: [NSAttributedString.Key: Any] = [
    .font: NSFont.systemFont(ofSize: 650, weight: .black),
    .foregroundColor: NSColor(calibratedRed: 0.173, green: 0.176, blue: 0.192, alpha: 1),
    .kern: -24,
]

("R" as NSString).draw(at: NSPoint(x: 243, y: 162), withAttributes: letterAttributes)

let accent = NSBezierPath()
accent.move(to: NSPoint(x: 512, y: 300))
accent.line(to: NSPoint(x: 636, y: 300))
accent.line(to: NSPoint(x: 782, y: 164))
accent.line(to: NSPoint(x: 658, y: 164))
accent.close()
NSColor(calibratedRed: 0.216, green: 0.404, blue: 0.957, alpha: 1).setFill()
accent.fill()

image.unlockFocus()

guard
    let tiff = image.tiffRepresentation,
    let bitmap = NSBitmapImageRep(data: tiff),
    let png = bitmap.representation(using: .png, properties: [:])
else {
    fatalError("Could not encode icon PNG")
}

try png.write(to: outputURL)
