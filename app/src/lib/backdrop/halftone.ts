// A print in halftone: a sun over the sea, in dots of ink on paper. The
// dots stand on a grid turned 45 degrees, as a printer's screen is; a dark
// tone is a large dot and a light one a small dot. The sun is printed in a
// second ink. It rises and sinks slowly, and the swell rolls under it. By
// night the paper is dark and the dots are pale.
import { put, type Frame, type Rgb, type Scene } from './craft.ts'

// the paper of the rail, ink, the second ink
const DAY: Rgb[] = [[242, 223, 206], [51, 48, 46], [153, 15, 61]]
const NIGHT: Rgb[] = [[38, 35, 33], [184, 170, 156], [181, 66, 104]]
const [PAPER, INK, SECOND] = [0, 1, 2]

// the screen: one dot every this many cells, and how large the darkest dot is
const PITCH = 7
const FULL = PITCH * 0.66
const TURN = Math.SQRT1_2

// how dark the print is at a point, and in which ink, inside the room between the rows
function toneAt(x: number, y: number, frame: Frame, time: number): [number, number] {
  const { cols, navRows, footRow } = frame
  const room = Math.max(1, footRow - navRows)
  const down = (y - navRows) / room
  const horizon = 0.62
  // the sun stands on the horizon and rises a little and sinks again, once in about two minutes
  const sunX = cols * 0.5
  const sunY = navRows + room * (horizon - 0.1 - 0.06 * Math.sin(time * 0.05))
  const sunR = Math.min(cols * 0.3, room * 0.12)
  if (down < horizon) {
    const fromSun = Math.hypot(x - sunX, y - sunY) / sunR
    if (fromSun < 1) return [0.8 - 0.25 * fromSun ** 2, SECOND]
    // the sky is light, and a little darker toward the top: dawn
    return [0.05 + 0.2 * (1 - down / horizon) ** 2, INK]
  }
  // Under the horizon the swell comes in in level bands that grow wider and
  // darker toward the foot, as water seen from the shore does. Where the
  // sun's light falls on a crest it is printed in the second ink.
  const below = (down - horizon) / (1 - horizon)
  const band = Math.sin((Math.sqrt(below) * 9 - time * 0.12) * Math.PI * 2 + Math.sin(x * 0.02 + time * 0.15) * 0.6)
  const crest = band > 0.35
  const path = crest && Math.abs(x - sunX) < sunR * (0.35 + 0.9 * below)
  if (path) return [0.35 + 0.25 * below, SECOND]
  return [crest ? 0.18 + 0.45 * below : 0.04 + 0.08 * below, INK]
}

function paint(frame: Frame, time: number, dark: boolean) {
  const colors = dark ? NIGHT : DAY
  const { cols, rows, open } = frame
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      const clear = open[y * cols + x]
      let place = PAPER
      if (clear > 0) {
        // the centre of the dot this cell belongs to, on the turned screen
        const [u, v] = [(x + y) * TURN, (x - y) * TURN]
        const [cu, cv] = [(Math.floor(u / PITCH) + 0.5) * PITCH, (Math.floor(v / PITCH) + 0.5) * PITCH]
        const [cx, cy] = [(cu + cv) * TURN, (cu - cv) * TURN]
        const [tone, ink] = toneAt(cx, cy, frame, time)
        const radius = FULL * Math.sqrt(Math.max(0, tone) * clear)
        if (Math.hypot(u - cu, v - cv) < radius) place = ink
      }
      put(frame, x, y, colors[place])
    }
  }
}

export const halftone: Scene = {
  colors: { light: DAY, dark: NIGHT },
  ground: { light: [PAPER], dark: [PAPER] },
  specks: { light: [], dark: [] },
  pace: 10,
  cell: 1,
  paint,
}
