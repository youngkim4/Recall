// The arithmetic of the small pictures in the Overview's tiles, and of the
// lines under them.

import { formatNumber } from './format.ts'

const round = (value: number) => Math.round(value * 100) / 100

// How many messages came in a day, over this many days, to the nearest one.
// Fewer than one a day says so: it is not none, and it is not one. No line
// when there are no days or no messages to speak of.
export function aDay(messages: number, days: number) {
  if (days <= 0 || messages <= 0) return ''
  const each = messages / days
  return each < 1 ? 'Under 1 a day' : `${formatNumber(Math.round(each))} a day`
}

// where each value sits: spread over the width, the largest at the top
function sparkPoints(values: number[], width: number, height: number): Array<[number, number]> {
  const lowest = Math.min(...values)
  const highest = Math.max(...values)
  const range = highest - lowest
  // values that are all the same lie along the middle, unless they are all nothing
  const level = highest > 0 ? height / 2 : height
  // a single value still makes a line, from one side to the other
  const spread = values.length === 1 ? [values[0], values[0]] : values
  return spread.map((value, index) => [
    round((index / (spread.length - 1)) * width),
    range > 0 ? round(height - ((value - Math.min(0, lowest)) / (highest - Math.min(0, lowest))) * height) : level,
  ])
}

// The path of a line through the values, for an SVG of this size.
export function sparkLine(values: number[], width: number, height: number) {
  if (!values.length) return ''
  return sparkPoints(values, width, height)
    .map(([x, y], index) => `${index ? 'L' : 'M'}${x} ${y}`)
    .join('')
}

// The same line closed down to the floor, to fill beneath it.
export function sparkArea(values: number[], width: number, height: number) {
  const line = sparkLine(values, width, height)
  return line ? `${line}L${width} ${height}L0 ${height}Z` : ''
}

// The largest parts as shares of the whole, largest first, at most `count`
// of them. The parts never come to more than the whole.
export function topShares(parts: number[], whole: number, count: number) {
  if (whole <= 0) return []
  return [...parts]
    .sort((a, b) => b - a)
    .slice(0, count)
    .reduce<number[]>((shares, part) => {
      const left = 1 - shares.reduce((sum, share) => sum + share, 0)
      return [...shares, round(Math.max(0, Math.min(left, part / whole)))]
    }, [])
}
