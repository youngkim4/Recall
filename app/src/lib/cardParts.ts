// The pieces a share card is put together from: who it is about, the figure
// it leads with, messages as the bubbles they were, the split of the two
// voices, and a few named figures. Each is a part with a height, which
// cardLayout places.

import type { Block } from './cardLayout.ts'
import {
  BETWEEN,
  COLOR,
  MARGIN,
  SPAN,
  WIDTH,
  clipped,
  fillRounded,
  fitSize,
  named,
  setType,
  wrap,
  type Pen,
} from './cardPen.ts'

export type CardQuote = {
  text: string
  // null when the export does not say who sent it
  fromMe: boolean | null
}

// a message with words in it, or null
export const spoken = (quote?: CardQuote | null) => (quote?.text?.trim() ? quote : null)

const BAR_HEIGHT = 24
const BAR_GAP = 6
// a word over a group of bubbles, and the room it takes
const OVER = 28
const OVER_ROOM = 48

// the conversation's name over a quieter line, the way Messages heads a thread
export function heading(pen: Pen, name: string, under: string, top: number) {
  setType(pen, fitSize(pen, name, 48, 600, SPAN), 600)
  pen.fillText(clipped(pen, name, SPAN), WIDTH / 2, top + 48)
  if (!under) return
  setType(pen, 32, 400, COLOR.quiet)
  pen.fillText(clipped(pen, under, SPAN), WIDTH / 2, top + 104)
}

export function wordmark(pen: Pen, base: number) {
  setType(pen, 30, 600, COLOR.mark)
  pen.fillText('Recall', WIDTH / 2, base)
}

export type HeroLine = { text: string; size: number; weight?: number; color?: string }

// The figure a card leads with, set large, over the words that say what it
// counts. A figure too wide for the card is set smaller.
export function hero(pen: Pen, numeral: string, size: number, lines: HeroLine[]): Block {
  const fitted = fitSize(pen, numeral, size, 600, SPAN)
  // figures stand about three quarters of their size tall
  const tall = fitted * 0.74
  const steps = lines.map((line, index) => line.size * (index === 0 ? 1.55 : 1.45))
  const last = lines[lines.length - 1]
  return {
    min: tall + steps.reduce((sum, step) => sum + step, 0) + (last ? last.size * 0.28 : 0),
    draw(to, top) {
      setType(to, fitted, 600)
      to.fillText(clipped(to, numeral, SPAN), WIDTH / 2, top + tall)
      lines.reduce((base, line, index) => {
        const next = base + steps[index]
        setType(to, line.size, line.weight ?? 500, line.color ?? COLOR.text)
        to.fillText(clipped(to, line.text, SPAN), WIDTH / 2, next)
        return next
      }, top + tall)
    },
  }
}

type Bubble = { quote: CardQuote; lines: string[]; width: number; height: number }

// one message measured: its lines, and the bubble that holds them
function measure(pen: Pen, quote: CardQuote, size: number, maxLines: number): Bubble {
  setType(pen, size, 400, COLOR.text, 'left')
  const lines = wrap(pen, quote.text.trim(), SPAN - size * 1.5, maxLines)
  const width = Math.max(0, ...lines.map((line) => pen.measureText(line).width)) + size * 1.5
  return { quote, lines, width, height: lines.length * size * 1.3 + size }
}

// One message as the bubble it was: yours in blue on the right, theirs in
// gray on the left, the corner nearest the speaker tightened into a tail.
function drawBubble(pen: Pen, { quote, lines, width, height }: Bubble, size: number, top: number) {
  const mine = quote.fromMe === true
  const x = mine ? WIDTH - MARGIN - width : MARGIN
  const round = Math.min(size * 1.1, height / 2)
  const tail = size * 0.3
  fillRounded(pen, mine ? COLOR.you : COLOR.them, x, top, width, height, mine ? [round, round, tail, round] : [round, round, round, tail])
  setType(pen, size, 400, mine ? COLOR.onYou : COLOR.onThem, 'left')
  lines.forEach((line, index) => pen.fillText(line, x + size * 0.75, top + size * 0.5 + size * 0.98 + index * size * 1.3))
}

// Messages one under another, each on its speaker's side, with a quiet word
// over them when one is given. Null when none of them has words in it.
export function messages(
  pen: Pen,
  quotes: Array<CardQuote | null | undefined>,
  type: { size: number; maxLines: number },
  over = '',
): Block | null {
  const said = quotes.map(spoken).filter((quote): quote is CardQuote => quote !== null)
  if (!said.length) return null
  const bubbles = said.map((quote) => measure(pen, quote, type.size, type.maxLines))
  const gap = type.size * 0.3
  const lead = over ? OVER_ROOM : 0
  return {
    min: lead + bubbles.reduce((sum, bubble) => sum + bubble.height, 0) + gap * (bubbles.length - 1),
    draw(to, top) {
      if (over) {
        setType(to, OVER, 400, COLOR.quiet)
        to.fillText(over, WIDTH / 2, top + OVER)
      }
      bubbles.reduce((at, bubble) => {
        drawBubble(to, bubble, type.size, at)
        return at + bubble.height + gap
      }, top + lead)
    },
  }
}

// How the full width is split into one bar for each voice. A voice that said
// nothing has no bar; one that said very little still gets a bar as wide as
// it is tall.
function barWidths(yours: number): [yours: number, theirs: number] {
  if (yours <= 0) return [0, SPAN]
  if (yours >= 1) return [SPAN, 0]
  const width = Math.max(BAR_HEIGHT, Math.min(SPAN - BAR_GAP - BAR_HEIGHT, yours * SPAN - BAR_GAP / 2))
  return [width, SPAN - BAR_GAP - width]
}

// one bar split between the two voices, like the bars inside the app, with each share named under its end
export function balance(yours: number, them: string): Block {
  return {
    min: 84,
    draw(pen, top) {
      const [yoursWidth, theirsWidth] = barWidths(yours)
      const end = BAR_HEIGHT / 2
      const join = 4
      if (yoursWidth) {
        fillRounded(pen, COLOR.you, MARGIN, top, yoursWidth, BAR_HEIGHT, theirsWidth ? [end, join, join, end] : [end, end, end, end])
      }
      if (theirsWidth) {
        const x = WIDTH - MARGIN - theirsWidth
        fillRounded(pen, COLOR.mark, x, top, theirsWidth, BAR_HEIGHT, yoursWidth ? [join, end, end, join] : [end, end, end, end])
      }
      // whole percents that add up to a hundred
      const yoursPercent = Math.round(yours * 100)
      const mine = `You ${yoursPercent}%`
      setType(pen, 32, 400, COLOR.quiet, 'left')
      pen.fillText(mine, MARGIN, top + 72)
      const room = SPAN - pen.measureText(mine).width - BETWEEN
      setType(pen, 32, 400, COLOR.quiet, 'right')
      pen.fillText(named(pen, them, ` ${100 - yoursPercent}%`, room), WIDTH - MARGIN, top + 72)
    },
  }
}

// a line of the card that names a figure; the value is fitted to the room its label leaves
export type Fact = { label: string; value: (pen: Pen, room: number) => string }

// Named figures, a line each between hairlines. Null when there are none.
export function facts(rows: Fact[], pitch: number, rank: number): Block | null {
  if (!rows.length) return null
  const rule = (pen: Pen, y: number) => {
    pen.fillStyle = COLOR.line
    pen.fillRect(MARGIN, y, SPAN, 2)
  }
  return {
    min: rows.length * pitch + 2,
    rank,
    draw(pen, top) {
      rows.forEach(({ label, value }, index) => {
        const line = top + index * pitch
        rule(pen, line)
        const base = line + pitch * 0.5 + 13
        setType(pen, 34, 400, COLOR.quiet, 'left')
        pen.fillText(label, MARGIN, base)
        const room = SPAN - pen.measureText(label).width - BETWEEN
        setType(pen, 34, 500, COLOR.text, 'right')
        pen.fillText(value(pen, room), WIDTH - MARGIN, base)
      })
      rule(pen, top + rows.length * pitch)
    },
  }
}
