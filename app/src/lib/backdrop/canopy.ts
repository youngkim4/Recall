// Under trees, looking up. Big split leaves reach in from the edges and
// sway, at three depths; one grows up through the empty middle of the rail.
// By day the leaves stand against a bright sky and the sun dapples them. By
// night they are lit from behind by the moon, and fireflies come and go.
import { chance, dithered, put, type Frame, type Rgb, type Scene } from './craft.ts'

// air, brighter air, far leaf, middle leaf, near leaf, the lit side of a near leaf, rib, fleck of light
const DAY: Rgb[] = [[230, 241, 214], [244, 249, 230], [190, 220, 176], [124, 184, 122], [52, 128, 80], [88, 162, 98], [28, 90, 60], [255, 253, 222]]
const NIGHT: Rgb[] = [[5, 16, 13], [8, 26, 20], [10, 36, 27], [14, 52, 36], [24, 82, 54], [36, 108, 68], [70, 150, 90], [240, 236, 130]]
const [FAR, MIDDLE, NEAR, LIT, RIB, FLECK] = [2, 3, 4, 5, 6, 7]

// A leaf, in cells: where its stem starts, how long it is, which way it
// points, how deep it hangs, and where in its sway it starts.
type Leaf = { x: number; y: number; size: number; turn: number; depth: number; sway: number }

// the leaves of one frame, far ones first: seven around the edges of the window and one in the rail
function leavesOf(frame: Frame): Leaf[] {
  const { cols, rows, railCols, navRows, footRow } = frame
  const short = Math.min(cols, rows)
  const around: Leaf[] = [
    { x: cols * 0.62, y: -short * 0.06, size: short * 0.5, turn: 1.9, depth: FAR, sway: 0 },
    { x: cols + short * 0.05, y: rows * 0.3, size: short * 0.62, turn: 2.75, depth: FAR, sway: 1.7 },
    { x: cols * 0.45, y: rows + short * 0.08, size: short * 0.5, turn: -1.2, depth: FAR, sway: 3.1 },
    { x: cols + short * 0.04, y: rows * 0.78, size: short * 0.56, turn: -2.7, depth: MIDDLE, sway: 0.9 },
    { x: cols * 0.86, y: -short * 0.05, size: short * 0.44, turn: 1.25, depth: MIDDLE, sway: 2.4 },
    { x: cols * 0.72, y: rows + short * 0.05, size: short * 0.46, turn: -1.85, depth: NEAR, sway: 4 },
    { x: cols + short * 0.03, y: rows * 0.04, size: short * 0.36, turn: 2.3, depth: NEAR, sway: 5.2 },
  ]
  // the leaf in the rail grows from the low corner, up into the room between the rows
  const room = footRow - navRows
  if (railCols < 12 || room < 8) return around
  return [...around, { x: -railCols * 0.08, y: footRow - room * 0.04, size: room * 0.92, turn: -1.02, depth: NEAR, sway: 2.9 }]
}

// where a split leaf is cut in from its edge: angles from the way it points
const CUTS = [0.48, 0.86, 1.28, 1.76, 2.3]

// Half the width of a leaf at each step along it, for a leaf of length 1. A
// heart: round at the stem, widest a third of the way along, drawn in to a point.
const STEPS = 512
const HALF = Float32Array.from({ length: STEPS + 1 }, (_, step) => {
  const along = step / STEPS
  return 0.5 * Math.pow(Math.sin(Math.PI * Math.pow(along, 0.58)), 0.72) * (1 - 0.3 * along)
})

// what a point on a leaf is: 0 off the leaf, 1 the blade, 2 a rib
function partOf(along: number, across: number): 0 | 1 | 2 {
  const half = HALF[Math.round(along * STEPS)]
  const side = Math.abs(across)
  if (side > half) return 0
  if (side < 0.011) return 2
  // each cut runs in from the edge and closes to a point before the middle;
  // a side rib runs out through the lobe between two cuts
  const angle = Math.atan2(side, along - 0.16)
  const reach = side / half
  for (const cut of CUTS) {
    const off = Math.abs(angle - cut)
    if (reach > 0.36 && off < 0.085 * (reach - 0.36)) return 0
    if (Math.abs(angle - cut + 0.2) < 0.009 / (0.2 + reach) && reach < 0.88) return 2
  }
  // a split leaf has holes beside its middle rib
  const hole = ((along - 0.52) / 0.06) ** 2 + ((side - 0.085) / 0.022) ** 2
  return hole < 1 ? 0 : 1
}

// what each cell of a frame shows, kept from one paint of that frame to the next
const SHOWN = new WeakMap<Frame, Uint8Array>()

function shownOf(frame: Frame): Uint8Array {
  const kept = SHOWN.get(frame) ?? new Uint8Array(frame.cols * frame.rows)
  SHOWN.set(frame, kept)
  return kept
}

function paint(frame: Frame, time: number, dark: boolean) {
  const colors = dark ? NIGHT : DAY
  const { cols, rows, open } = frame
  // what each cell shows: first the air, with light coming down in slow slanted shafts
  const shown = shownOf(frame)
  const shaftAt = (x: number, y: number) => 0.5 + 0.5 * Math.sin((x * 1.3 + y) * 0.028 - time * 0.1)
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) shown[y * cols + x] = dithered(shaftAt(x, y) * 1.6 - 0.3, x, y, 1)
  }
  // then each leaf over it, within the square of cells it can reach
  for (const leaf of leavesOf(frame)) {
    // each leaf turns a little about its stem
    const turn = leaf.turn + 0.06 * Math.sin(time * 0.33 + leaf.sway) + 0.02 * Math.sin(time * 0.9 + leaf.sway * 2)
    const [cos, sin] = [Math.cos(turn) / leaf.size, Math.sin(turn) / leaf.size]
    const [fromX, toX] = [Math.max(0, Math.floor(leaf.x - leaf.size)), Math.min(cols - 1, Math.ceil(leaf.x + leaf.size))]
    const [fromY, toY] = [Math.max(0, Math.floor(leaf.y - leaf.size)), Math.min(rows - 1, Math.ceil(leaf.y + leaf.size))]
    for (let y = fromY; y <= toY; y += 1) {
      for (let x = fromX; x <= toX; x += 1) {
        const [dx, dy] = [x - leaf.x, y - leaf.y]
        const along = dx * cos + dy * sin
        if (along <= 0 || along >= 1 || open[y * cols + x] === 0) continue
        const across = dy * cos - dx * sin
        if (across > 0.5 || across < -0.5) continue
        const part = partOf(along, across)
        if (!part) continue
        // the side of a near leaf that a shaft falls on is a step brighter
        const lit = leaf.depth === NEAR && dithered(shaftAt(x, y) * 2.2 - 0.9, x, y, 1) === 1
        shown[y * cols + x] = part === 2 ? (leaf.depth === NEAR ? RIB : leaf.depth + 1) : lit ? LIT : leaf.depth
      }
    }
  }
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      let place = shown[y * cols + x]
      // sun through the leaves, or fireflies: each lives in its own cell and comes and goes
      const life = chance(x, y, 3)
      if (open[y * cols + x] === 1 && life > (dark ? 0.9955 : 0.985)) {
        const glow = Math.sin(time * (dark ? 0.9 : 0.4) + life * 900)
        if (dark ? glow > 0.5 : place >= FAR && shaftAt(x, y) > 0.55 && glow > 0.2) place = FLECK
      }
      put(frame, x, y, colors[place])
    }
  }
}

export const canopy: Scene = { colors: { light: DAY, dark: NIGHT }, ground: { light: [0, 1], dark: [0, 1] }, specks: { light: [7], dark: [7] }, pace: 6, paint }
