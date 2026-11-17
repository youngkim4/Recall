// What every part of a share card is drawn with. A card is an object that
// leaves the app, so it carries its own fixed colors, the ones Messages uses
// in light mode, whatever theme the app is in. Text is fitted, wrapped or cut
// here, so that nothing a card says leaves its margins.

export const WIDTH = 1080
export const MARGIN = 96
// the width between the margins: nothing is drawn outside it
export const SPAN = WIDTH - MARGIN * 2
// below this a name no longer reads on a card; one still too wide is cut instead
const SMALLEST_NAME = 40
// the least room between two pieces of text that share a line
export const BETWEEN = 32
const FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Arial, sans-serif'

export const COLOR = {
  paper: '#ffffff',
  text: '#1d1d1f',
  quiet: '#636368',
  line: 'rgba(60, 60, 67, 0.18)',
  // the line a chart stands on
  ground: 'rgba(60, 60, 67, 0.36)',
  you: '#0071e3',
  onYou: '#ffffff',
  them: '#e9e9eb',
  mark: '#8e8e93',
  // a bar that is there for the others to be read against
  faint: '#d1d1d6',
}

export type Pen = CanvasRenderingContext2D
export type Corners = [topLeft: number, topRight: number, bottomRight: number, bottomLeft: number]

export const numbers = new Intl.NumberFormat()
// what the end of a chart's scale stands for: "2.5K"
export const compact = new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 })
export const count = (value: number, word: string) => `${numbers.format(value)} ${value === 1 ? word : `${word}s`}`
export const percent = (share: number) => `${Math.round(share * 100)}%`
// a share from 0 to 1, or null when the figure is missing or is not a number
export const shareOf = (value?: number | null) =>
  typeof value === 'number' && Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : null

export function setType(pen: Pen, size: number, weight = 400, color = COLOR.text, align: CanvasTextAlign = 'center') {
  pen.font = `${weight} ${size}px ${FONT}`
  pen.fillStyle = color
  pen.textAlign = align
  pen.textBaseline = 'alphabetic'
}

// the largest size, from `size` down to the smallest a name is set in, at which the text fits the width
export function fitSize(pen: Pen, text: string, size: number, weight: number, maxWidth: number) {
  const smallest = Math.min(size, SMALLEST_NAME)
  const sizes = Array.from({ length: Math.floor((size - smallest) / 8) + 1 }, (_, step) => size - step * 8)
  return (
    sizes.find((candidate) => {
      pen.font = `${weight} ${candidate}px ${FONT}`
      return pen.measureText(text).width <= maxWidth
    }) ?? smallest
  )
}

// The text with an ellipsis after it, cut back a letter at a time until both
// fit the width. The pen is already set to the type the text is drawn in.
function trailOff(pen: Pen, text: string, maxWidth: number) {
  const letters = [...text]
  const ending = (kept: number) => `${letters.slice(0, kept).join('').replace(/[.,!?\s]+$/, '')}…`
  const kept = Array.from({ length: letters.length + 1 }, (_, dropped) => letters.length - dropped).find(
    (length) => pen.measureText(ending(length)).width <= maxWidth,
  )
  return ending(kept ?? 0)
}

// the text as it is when it fits the width, trailing off when it does not
export function clipped(pen: Pen, text: string, maxWidth: number) {
  return pen.measureText(text).width <= maxWidth ? text : trailOff(pen, text, maxWidth)
}

// a name with what is said of it: the name gives way, so the rest is never cut
export function named(pen: Pen, name: string, rest: string, maxWidth: number) {
  return `${clipped(pen, name, maxWidth - pen.measureText(rest).width)}${rest}`
}

// a word wider than the line is cut into pieces that fit, so nothing leaves the card
function pieces(pen: Pen, word: string, maxWidth: number): string[] {
  if (pen.measureText(word).width <= maxWidth) return [word]
  return [...word].reduce<string[]>((parts, letter) => {
    const last = parts[parts.length - 1] ?? ''
    return last && pen.measureText(last + letter).width <= maxWidth
      ? [...parts.slice(0, -1), last + letter]
      : [...parts, letter]
  }, [])
}

// The text in lines no wider than maxWidth. Past maxLines the last line trails off.
export function wrap(pen: Pen, text: string, maxWidth: number, maxLines: number): string[] {
  const words = text
    .split(/\s+/)
    .filter(Boolean)
    .flatMap((word) => pieces(pen, word, maxWidth))
  const lines = words.reduce<string[]>((done, word) => {
    const last = done[done.length - 1]
    if (last !== undefined && pen.measureText(`${last} ${word}`).width <= maxWidth) {
      return [...done.slice(0, -1), `${last} ${word}`]
    }
    return [...done, word]
  }, [])
  if (lines.length <= maxLines) return lines
  if (maxLines < 1) return []
  return [...lines.slice(0, maxLines - 1), trailOff(pen, lines[maxLines - 1], maxWidth)]
}

// The path of a rectangle with each corner rounded by its own radius. A
// canvas with no roundRect of its own gets the same path drawn by hand.
export function roundedRect(pen: Pen, x: number, y: number, width: number, height: number, corners: Corners) {
  const [topLeft, topRight, bottomRight, bottomLeft] = corners.map((radius) =>
    Math.max(0, Math.min(radius, width / 2, height / 2)),
  )
  pen.beginPath()
  if (typeof pen.roundRect === 'function') {
    pen.roundRect(x, y, width, height, [topLeft, topRight, bottomRight, bottomLeft])
    return
  }
  pen.moveTo(x + topLeft, y)
  pen.arcTo(x + width, y, x + width, y + height, topRight)
  pen.arcTo(x + width, y + height, x, y + height, bottomRight)
  pen.arcTo(x, y + height, x, y, bottomLeft)
  pen.arcTo(x, y, x + width, y, topLeft)
  pen.closePath()
}

// a filled shape in one call
export function fillRounded(pen: Pen, color: string, x: number, y: number, width: number, height: number, corners: Corners) {
  pen.fillStyle = color
  roundedRect(pen, x, y, width, height, corners)
  pen.fill()
}
