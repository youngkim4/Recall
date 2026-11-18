// A small lamp drawn in cells, for the reasoning control: the more the
// model is asked to think, the taller and brighter its flame. At the last
// step it burns full and sparks rise from it. The flame is the two colours
// of the theme's scale, its heart the far colour made white-hot. Nothing
// here touches the page: it fills a list of cells, and a canvas shows it.

export type Rgb = readonly [number, number, number]

// the colours a lamp is drawn in: the candle, its edge and wick, and the three of the flame
export type LampColors = { wax: Rgb; edge: Rgb; outer: Rgb; middle: Rgb; heart: Rgb }

export type LampFrame = { cells: Uint8ClampedArray<ArrayBuffer>; cols: number; rows: number }

// As tall as the name and the row of marks beside it (slider.css), so an
// unlit candle leaves no empty band over the words.
export const LAMP = { cols: 22, rows: 20 }

// where the candle stands, in cells: a short one, like a votive, on a dish
// that is wider than it, in the bottom row
const CANDLE = { left: 8, right: 13, top: 13 }
const DISH = { left: 4, right: 17 }
export const WICK_ROW = CANDLE.top - 1
// the flame on the first step and the last, in cells over the wick
const LOW = 4
const HIGH = 8
// how far sparks and smoke rise over where they start, in cells
const SPARK_RISE = 4
const SMOKE_RISE = 7

export function lampFrame(): LampFrame {
  return { cells: new Uint8ClampedArray(LAMP.cols * LAMP.rows * 4), cols: LAMP.cols, rows: LAMP.rows }
}

function put(frame: LampFrame, x: number, y: number, color: Rgb, alpha = 255) {
  if (x < 0 || y < 0 || x >= frame.cols || y >= frame.rows) return
  const at = (y * frame.cols + x) * 4
  frame.cells[at] = color[0]
  frame.cells[at + 1] = color[1]
  frame.cells[at + 2] = color[2]
  frame.cells[at + 3] = alpha
}

// a number from 0 to 1 that is always the same for the same arguments
function chance(a: number, b: number) {
  const mixed = Math.sin(a * 127.1 + b * 311.7) * 43758.5453
  return mixed - Math.floor(mixed)
}

// how tall the flame is, in cells: by the step, a flicker, and a flare just after a step
export function flameHeight(level: number, time: number, flare: number) {
  const flicker = 0.5 * Math.sin(time * 9) + 0.35 * Math.sin(time * 23 + 1.3)
  return LOW + (HIGH - LOW) * level + flicker * (0.4 + level * 0.6) + flare * 2.5
}

// The candle, its wick and its dish, drawn last so the flame stands behind
// none of them. At this size an outline reads as a little table, so the
// candle is solid, with one stripe of light down its side.
function paintCandle(frame: LampFrame, middleX: number, colors: LampColors) {
  const dish = frame.rows - 1
  put(frame, Math.round(middleX - 0.5), WICK_ROW, colors.edge)
  for (let y = CANDLE.top; y < dish; y += 1) {
    for (let x = CANDLE.left; x <= CANDLE.right; x += 1) {
      const shine = x === CANDLE.left + 1 && y > CANDLE.top
      put(frame, x, y, shine ? colors.wax : colors.edge)
    }
  }
  for (let x = DISH.left; x <= DISH.right; x += 1) put(frame, x, dish, colors.edge)
}

// An unlit candle: a thin wisp of smoke rises from the wick and drifts.
function paintSmoke(frame: LampFrame, middleX: number, time: number, colors: LampColors) {
  for (let rise = 1; rise <= SMOKE_RISE; rise += 1) {
    const drift = Math.sin(time * 1.6 + rise * 0.55) * (rise / 4)
    put(frame, Math.round(middleX - 0.5 + drift), WICK_ROW - rise, colors.edge, Math.round(150 * (1 - rise / (SMOKE_RISE + 1))))
  }
}

// Paint one frame. `level` runs 0 to 1 over the steps; `max` is the last
// step; `lit` is false on a step that asks for no thinking at all, where the
// candle is out; `time` in seconds; `flare` 0 to 1, how much of a fresh
// step's flare is left. Cells with nothing on them stay clear.
export function paintLamp(
  frame: LampFrame,
  level: number,
  max: boolean,
  time: number,
  flare: number,
  colors: LampColors,
  lit = true,
) {
  frame.cells.fill(0)
  const middleX = (CANDLE.left + CANDLE.right) / 2
  if (!lit) {
    paintSmoke(frame, middleX, time, colors)
    paintCandle(frame, middleX, colors)
    return
  }
  const height = flameHeight(level, time, flare)
  const tip = WICK_ROW - height
  const wide = 2.2 + level * 2
  // the tip leans with the air, more when the flame is tall
  const lean = Math.sin(time * 5.3) * (0.4 + level * 0.8)

  // the glow: a soft light of the far colour around the flame, wider and
  // brighter with each step, strongest at the last
  const glow = 0.25 + level * 0.45 + (max ? 0.3 : 0)
  const glowY = tip + height * 0.6
  const reach = height * 0.85 + 3
  for (let y = 0; y < frame.rows; y += 1) {
    for (let x = 0; x < frame.cols; x += 1) {
      const fromFlame = Math.hypot((x + 0.5 - middleX) / 1.15, y + 0.5 - glowY) / reach
      if (fromFlame < 1) put(frame, x, y, colors.middle, Math.round(glow * (1 - fromFlame) ** 2 * 90))
    }
  }

  // the flame: a drop, widest a third of the way up, in three bands of
  // colour with no dither: the theme's hue at its edge, the far colour
  // inside, and a white-hot heart low over the wick
  for (let y = Math.floor(tip); y <= WICK_ROW; y += 1) {
    const up = (WICK_ROW - y) / Math.max(1, height)
    if (up < 0 || up > 1) continue
    const half = wide * Math.sin(Math.PI * Math.pow(1 - up, 0.65)) * (1 - up * 0.3)
    const centre = middleX + lean * up * up
    for (let x = Math.floor(centre - half - 1); x <= Math.ceil(centre + half + 1); x += 1) {
      const across = Math.abs(x + 0.5 - centre) / Math.max(0.55, half)
      if (across > 1) continue
      const heat = (1 - across) * (1 - up * 0.75)
      put(frame, x, y, heat > 0.5 ? colors.heart : heat > 0.22 ? colors.middle : colors.outer)
    }
  }

  // at the last step sparks rise from the tip and go out
  if (max) {
    for (let spark = 0; spark < 5; spark += 1) {
      const life = (time * 0.7 + spark * 0.21 + chance(spark, 3) * 0.3) % 1
      if (life > 0.85) continue
      const y = Math.round(tip - 1 - life * SPARK_RISE)
      const x = Math.round(middleX + Math.sin(spark * 2.3 + life * 5) * (1 + life * 2.5))
      put(frame, x, y, life < 0.35 ? colors.heart : colors.middle)
    }
  }

  paintCandle(frame, middleX, colors)
}
