// The charts a share card can carry, in the app's own chart language: what
// you sent rises above a line in blue and what you received hangs below it
// in gray; a count of both voices takes no voice's color. A chart on a card
// names its ends and its scale, so it can be read with nothing else beside it.

import { roundUp } from './activity.ts'
import type { DaysChart, VoicesChart } from './cardData.ts'
import type { Block } from './cardLayout.ts'
import { BETWEEN, COLOR, MARGIN, SPAN, WIDTH, clipped, compact, fillRounded, setType, type Pen } from './cardPen.ts'

// the room at the right of a chart for what its top and its bottom stand for
const SCALE = 96
const PLOT = SPAN - SCALE
const LABEL = 28
const SCALE_LABEL = 26
const HAIR = 2
const SWATCH = 20
// the room over a plot for its legend, and under it for the names of its ends
const LEGEND_ROOM = 56
const AXIS_ROOM = 52

type Sizes = { min: number; max: number; rank: number }

const most = (values: number[]) => values.reduce((top, value) => Math.max(top, value), 0)

function hairline(pen: Pen, color: string, y: number, width: number) {
  pen.fillStyle = color
  pen.fillRect(MARGIN, y, width, HAIR)
}

// what the end of a scale stands for, at the right of the chart
function scaleLabel(pen: Pen, value: number, base: number) {
  setType(pen, SCALE_LABEL, 400, COLOR.quiet, 'right')
  pen.fillText(compact.format(value), WIDTH - MARGIN, base)
}

// the names of the first and the last bar, under the two ends of the plot
function ends(pen: Pen, first: string, last: string, base: number) {
  setType(pen, LABEL, 400, COLOR.quiet, 'left')
  if (first) pen.fillText(first, MARGIN, base)
  setType(pen, LABEL, 400, COLOR.quiet, 'right')
  if (last) pen.fillText(last, MARGIN + PLOT, base)
}

// whose bar is which, and what a bar counts
function legend(pen: Pen, them: string, unit: string, base: number) {
  const swatch = (color: string, x: number) =>
    fillRounded(pen, color, x, base - SWATCH, SWATCH, SWATCH, [5, 5, 5, 5])
  setType(pen, LABEL, 400, COLOR.quiet, 'left')
  swatch(COLOR.you, MARGIN)
  setType(pen, LABEL, 400, COLOR.quiet, 'left')
  pen.fillText('You', MARGIN + SWATCH + 12, base)
  const next = MARGIN + SWATCH + 12 + pen.measureText('You').width + LABEL
  swatch(COLOR.mark, next)
  setType(pen, LABEL, 400, COLOR.quiet, 'left')
  const room = WIDTH - MARGIN - pen.measureText(unit).width - BETWEEN - (next + SWATCH + 12)
  pen.fillText(clipped(pen, them, room), next + SWATCH + 12, base)
  setType(pen, LABEL, 400, COLOR.quiet, 'right')
  pen.fillText(unit, WIDTH - MARGIN, base)
}

// The two voices over time. Each side has its own scale, as in the app, so a
// quiet voice is still read; the round number each side goes up to is named
// at its end.
export function voices(chart: VoicesChart, them: string, { min, max, rank }: Sizes): Block {
  return {
    min,
    max,
    rank,
    draw(pen, top, height) {
      const plotTop = top + LEGEND_ROOM
      const plotBottom = top + height - AXIS_ROOM
      const above = roundUp(most(chart.points.map((point) => point.sent)))
      const below = roundUp(most(chart.points.map((point) => point.received)))
      if (above + below === 0) return
      const ground = plotTop + (plotBottom - plotTop) * (above / (above + below))
      const slot = PLOT / chart.points.length
      const width = Math.max(2, Math.min(44, slot * 0.68))
      const round = Math.min(6, width / 2)

      legend(pen, them, chart.unit === 'month' ? 'messages a month' : 'messages a day', top + LABEL)
      if (above) {
        hairline(pen, COLOR.line, plotTop, SPAN)
        scaleLabel(pen, above, plotTop + SCALE_LABEL + 6)
      }
      if (below) {
        hairline(pen, COLOR.line, plotBottom - HAIR, SPAN)
        scaleLabel(pen, below, plotBottom - 12)
      }
      chart.points.forEach((point, index) => {
        const x = MARGIN + slot * index + (slot - width) / 2
        if (point.sent > 0) {
          const tall = Math.max(3, (point.sent / above) * (ground - plotTop - HAIR))
          fillRounded(pen, COLOR.you, x, ground - HAIR / 2 - tall, width, tall, [round, round, 0, 0])
        }
        if (point.received > 0) {
          const tall = Math.max(3, (point.received / below) * (plotBottom - ground - HAIR))
          fillRounded(pen, COLOR.mark, x, ground + HAIR / 2, width, tall, [0, 0, round, round])
        }
      })
      hairline(pen, COLOR.ground, ground - HAIR / 2, PLOT)
      ends(pen, chart.first, chart.last, top + height - 8)
    },
  }
}

// One day among the days around it: a bar a day, the day itself in ink, and
// the round number the tallest of them stands under named at the top.
export function days(chart: DaysChart, { min, max, rank }: Sizes): Block {
  return {
    min,
    max,
    rank,
    draw(pen, top, height) {
      const plotTop = top + 8
      const plotBottom = top + height - AXIS_ROOM
      const peak = roundUp(most(chart.counts)) || 1
      const slot = PLOT / chart.counts.length
      const width = Math.max(3, Math.min(40, slot * 0.68))
      const round = Math.min(6, width / 2)
      const center = (index: number) => MARGIN + slot * (index + 0.5)

      hairline(pen, COLOR.line, plotTop, SPAN)
      scaleLabel(pen, peak, plotTop + SCALE_LABEL + 6)
      chart.counts.forEach((held, index) => {
        // a day with nothing on it is a mark on the line, so its place is still there
        const tall = held > 0 ? Math.max(4, (held / peak) * (plotBottom - plotTop - HAIR)) : 3
        const color = index === chart.at ? COLOR.text : COLOR.faint
        fillRounded(pen, color, center(index) - width / 2, plotBottom - tall, width, tall, [round, round, 0, 0])
      })
      hairline(pen, COLOR.ground, plotBottom, PLOT)

      // the day is named under its own bar, and an end's name gives way to it
      const base = top + height - 8
      setType(pen, LABEL, 600, COLOR.text, 'left')
      const named = pen.measureText(chart.day).width
      const left = Math.max(MARGIN, Math.min(MARGIN + PLOT - named, center(chart.at) - named / 2))
      pen.fillText(chart.day, left, base)
      setType(pen, LABEL, 400, COLOR.quiet, 'left')
      const firstFits = MARGIN + pen.measureText(chart.first).width + BETWEEN / 2 <= left
      const lastFits = MARGIN + PLOT - pen.measureText(chart.last).width - BETWEEN / 2 >= left + named
      ends(pen, firstFits ? chart.first : '', lastFits ? chart.last : '', base)
    },
  }
}
