import assert from 'node:assert/strict'
import { test } from 'node:test'
import { LAMP, WICK_ROW, flameHeight, lampFrame, paintLamp, type LampColors } from './lamp.ts'

const colors: LampColors = {
  wax: [230, 225, 215],
  edge: [90, 85, 80],
  outer: [31, 74, 122],
  middle: [27, 127, 140],
  heart: [180, 230, 235],
}
const flameColors = [colors.outer, colors.middle, colors.heart]
const same = (a: readonly number[], b: readonly number[]) => a[0] === b[0] && a[1] === b[1] && a[2] === b[2]

// the highest row with a solid cell of the flame, and how many solid flame cells there are
function measure(level: number, max: boolean, time: number) {
  const frame = lampFrame()
  paintLamp(frame, level, max, time, 0, colors)
  let top = LAMP.rows
  let count = 0
  for (let y = 0; y < LAMP.rows; y += 1) {
    for (let x = 0; x < LAMP.cols; x += 1) {
      const at = (y * LAMP.cols + x) * 4
      const cell = [frame.cells[at], frame.cells[at + 1], frame.cells[at + 2]]
      if (frame.cells[at + 3] === 255 && flameColors.some((color) => same(color, cell))) {
        top = Math.min(top, y)
        count += 1
      }
    }
  }
  return { top, count, frame }
}

test('the flame is taller and larger on a higher step', () => {
  const low = measure(0, false, 1)
  const high = measure(1, false, 1)
  assert.ok(high.top < low.top, `top ${high.top} against ${low.top}`)
  assert.ok(high.count > low.count * 1.5, `cells ${high.count} against ${low.count}`)
})

test('the height of the flame moves a little with time, and more just after a step', () => {
  const heights = [0.1, 0.4, 0.7, 1.3].map((time) => flameHeight(0.5, time, 0))
  assert.ok(Math.max(...heights) - Math.min(...heights) > 0.2)
  assert.ok(flameHeight(0.5, 1, 1) > flameHeight(0.5, 1, 0) + 2)
})

test('the tallest flame, flicker and all, stays inside the frame, with room over it', () => {
  for (let time = 0; time < 6; time += 0.05) {
    assert.ok(WICK_ROW - flameHeight(1, time, 0) >= 2, `tip at ${WICK_ROW - flameHeight(1, time, 0)}`)
  }
})

test('sparks rise over the flame only on the last step', () => {
  const sparksOver = (max: boolean) => {
    let found = 0
    for (const time of [0.2, 0.5, 0.9, 1.4, 2.1]) {
      const { frame } = measure(1, max, time)
      // the rows over the tip of the flame at that moment
      const tip = Math.floor(WICK_ROW - flameHeight(1, time, 0))
      for (let y = 0; y < tip; y += 1) {
        for (let x = 0; x < LAMP.cols; x += 1) if (frame.cells[(y * LAMP.cols + x) * 4 + 3] === 255) found += 1
      }
    }
    return found
  }
  assert.equal(sparksOver(false), 0)
  assert.ok(sparksOver(true) > 0)
})

test('the candle stands on every step, and the frame holds only the colours of the lamp', () => {
  const { frame } = measure(0, false, 0)
  const all = [colors.wax, colors.edge, ...flameColors]
  let candle = 0
  for (let at = 0; at < frame.cells.length; at += 4) {
    if (frame.cells[at + 3] === 0) continue
    const cell = [frame.cells[at], frame.cells[at + 1], frame.cells[at + 2]]
    assert.ok(all.some((color) => same(color, cell)), `a colour not of the lamp: ${cell}`)
    if (same(cell, colors.wax) || same(cell, colors.edge)) candle += 1
  }
  assert.ok(candle > 40)
})

test('a candle that is out has no flame, only a wisp of smoke', () => {
  const frame = lampFrame()
  paintLamp(frame, 0, false, 1, 0, colors, false)
  let flame = 0
  let smoke = 0
  for (let at = 0; at < frame.cells.length; at += 4) {
    const cell = [frame.cells[at], frame.cells[at + 1], frame.cells[at + 2]]
    if (frame.cells[at + 3] > 0 && flameColors.some((color) => same(color, cell))) flame += 1
    if (frame.cells[at + 3] > 0 && frame.cells[at + 3] < 255 && same(cell, colors.edge)) smoke += 1
  }
  assert.equal(flame, 0)
  assert.ok(smoke > 3)
})
