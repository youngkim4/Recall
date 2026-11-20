// The pictures a share card and Replay first words stand on: three still
// scenes in the colors of the rail's own scenes (backdrop/), dithered on a
// coarse grid of cells the way the rail's are. A scene is a function from a
// cell to a color, so the card, its small swatch and the replay all draw the
// same picture at their own size.

import { chance, dithered, type Rgb } from './backdrop/craft.ts'
import { DAY as SEA_DAY, NIGHT as SEA_NIGHT } from './backdrop/shoreline.ts'
import { DAY as SKY_DAY, NIGHT as SKY_NIGHT } from './backdrop/sky.ts'

export type SceneName = 'sunrise' | 'sea' | 'aurora'

export const SCENES: ReadonlyArray<{ value: SceneName; label: string }> = [
  { value: 'sunrise', label: 'Sunrise' },
  { value: 'sea', label: 'Moonlit sea' },
  { value: 'aurora', label: 'Northern lights' },
]

const hex = ([r, g, b]: Rgb) => `#${[r, g, b].map((value) => value.toString(16).padStart(2, '0')).join('')}`

// The night the words stand on under each picture: the open water under the
// sunrise, the deep water under the moon, the zenith over the lights.
export const GROUND: Record<SceneName, string> = {
  sunrise: hex(SEA_NIGHT[1]),
  sea: hex(SEA_NIGHT[0]),
  aurora: hex(SKY_NIGHT[0]),
}

// the scene a card of each kind starts on
export function sceneFor(kind: 'anniversary' | 'onthisday' | 'duo'): SceneName {
  return kind === 'anniversary' ? 'sunrise' : kind === 'onthisday' ? 'sea' : 'aurora'
}

export type Painter = (x: number, y: number) => Rgb

// how rare a star is: a cell is one when its chance is over this
const STARS = 0.993

function star(x: number, y: number): Rgb | null {
  if (chance(x, y, 3) <= STARS) return null
  return chance(x, y, 4) > 0.5 ? SKY_NIGHT[6] : SKY_NIGHT[5]
}

// a low sun over the sea, a band of cloud, and the sun's path on the water
function sunrise(cols: number, rows: number): Painter {
  const horizon = Math.floor(rows * 0.66)
  const sunX = cols * 0.3
  const sunR = Math.max(4, cols * 0.11)
  return (x, y) => {
    if (y < horizon) {
      if (Math.hypot(x - sunX, y - (horizon - 1)) < sunR) return SKY_DAY[6]
      const cloud = 0.5 + 0.5 * Math.sin(x * 0.12 + y * 0.6) * Math.sin(x * 0.05 - 1)
      if (Math.abs(y - rows * 0.3) < rows * 0.06 && cloud > 0.55)
        return dithered((cloud - 0.55) * 6, x, y, 1) ? SKY_DAY[5] : SKY_DAY[4]
      return SKY_DAY[dithered((y / horizon) * 3.2, x, y, 3)]
    }
    const down = (y - horizon) / Math.max(1, rows - horizon)
    if (Math.abs(x - sunX) < sunR * (1 - down * 0.5) && (y - horizon) % 2 === 0 && chance(x, y, 1) > 0.4)
      return SKY_DAY[6]
    return SEA_DAY[2 - dithered(down * 2.4 + 0.25 * Math.sin(x * 0.3 + y), x, y, 2)]
  }
}

// the moon over a night sea, and its broken path on the water
function sea(cols: number, rows: number): Painter {
  const horizon = Math.floor(rows * 0.6)
  const moonX = cols * 0.7
  const moonY = rows * 0.25
  const moonR = Math.max(3, cols * 0.07)
  return (x, y) => {
    if (y < horizon) {
      const far = Math.hypot(x - moonX, y - moonY) / moonR
      if (far < 1) return SKY_NIGHT[6]
      if (far < 1.6 && dithered((1.6 - far) * 1.6, x, y, 1)) return SKY_NIGHT[5]
      return star(x, y) ?? SKY_NIGHT[dithered((y / horizon) * 2.2, x, y, 2)]
    }
    const down = (y - horizon) / Math.max(1, rows - horizon)
    const onPath = Math.abs(x - moonX) < 1.5 + down * cols * 0.08 && chance(x, y, 9) > 0.62 - 0.25 * (1 - down)
    if (onPath && (y - horizon) % 3 !== 0) return SEA_NIGHT[3]
    return SEA_NIGHT[2 - dithered(down * 2.4 + 0.25 * Math.sin(x * 0.35 + y * 1.3) + 0.2, x, y, 2)]
  }
}

// a curtain of northern light in rays, over stars; `base` is how far down it hangs
function aurora(rows: number, base = 0.32): Painter {
  return (x, y) => {
    const wave = rows * (base + 0.1 * Math.sin(x * 0.09) + 0.05 * Math.sin(x * 0.23 + 1))
    const into = (y - wave) / (rows * 0.22)
    if (into > -0.2 && into < 1) {
      const rays = 0.45 + 0.55 * Math.abs(Math.sin(x * 0.55 + Math.sin(x * 0.13) * 2))
      const strength = (into > 0 ? 1 - into : 1 + into * 4) * rays
      if (dithered(strength * 1.3, x, y, 1)) return into > 0.25 || chance(x, y, 2) > 0.3 ? SKY_NIGHT[3] : SKY_NIGHT[4]
    }
    return star(x, y) ?? SKY_NIGHT[dithered((y / rows) * 2.4, x, y, 2)]
  }
}

// The picture of a scene `cols` cells wide and `rows` tall. `curtain` is how
// far down the northern lights hang, as a share of the height: the replay
// hangs them high, to leave the middle of the window clear for its words.
export function painter(name: SceneName, cols: number, rows: number, curtain = 0.32): Painter {
  if (name === 'sunrise') return sunrise(cols, rows)
  if (name === 'sea') return sea(cols, rows)
  return aurora(rows, curtain)
}

type Ink = {
  fillStyle: string | CanvasGradient | CanvasPattern
  fillRect: (x: number, y: number, w: number, h: number) => void
}

// Draws a scene cell by cell, a run of one color as one rectangle.
export function paintScene(
  pen: Ink,
  name: SceneName,
  cols: number,
  rows: number,
  cell: number,
  at = { x: 0, y: 0 },
  curtain?: number,
) {
  const color = painter(name, cols, rows, curtain)
  for (let y = 0; y < rows; y += 1) {
    let start = 0
    let run = color(0, y)
    for (let x = 1; x <= cols; x += 1) {
      const next = x < cols ? color(x, y) : null
      if (next && next === run) continue
      pen.fillStyle = hex(run)
      pen.fillRect(at.x + start * cell, at.y + y * cell, (x - start) * cell, cell)
      if (next) {
        start = x
        run = next
      }
    }
  }
}
