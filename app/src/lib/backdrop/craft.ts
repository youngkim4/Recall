// The few tools every backdrop is made with. A backdrop is a picture of a
// few flat colours on a coarse grid of cells, the way a mosaic or a printed
// pattern is: a step between two colours is not blended, it is dithered.

export type Rgb = readonly [number, number, number]

// One picture's worth of cells, and where the rail's rows stand on it. A
// scene is composed around the rail: its rows are read against the picture
// with no glass between, so under them a scene paints its ground and
// nothing else. The empty middle of the rail is open, like the rest.
export type Frame = {
  // one byte each of red, green, blue and alpha for every cell, row by row
  cells: Uint8ClampedArray<ArrayBuffer>
  cols: number
  rows: number
  // the rail is this many cells wide; its upper rows end at navRows and its foot starts at footRow
  railCols: number
  navRows: number
  footRow: number
  // for every cell: 0 where a row of the rail stands, 1 well away from one, a slope between
  open: Float32Array
}

// how many cells from a row of the rail a scene is fully itself again, for cells of 4px
export const FEATHER = 9

function away(from: number, to: number, at: number): number {
  return at < from ? from - at : at >= to ? at - to + 1 : 0
}

export function frameOf(cols: number, rows: number, railCols: number, navRows: number, footRow: number, feather = FEATHER): Frame {
  const open = new Float32Array(cols * rows)
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      const across = away(0, railCols, x)
      const near = Math.min(Math.hypot(across, away(0, navRows, y)), Math.hypot(across, away(footRow, rows, y)))
      const share = railCols > 0 ? Math.min(1, near / feather) : 1
      open[y * cols + x] = share * share * (3 - 2 * share)
    }
  }
  return { cells: new Uint8ClampedArray(cols * rows * 4), cols, rows, railCols, navRows, footRow, open }
}

// a 4 by 4 ordered dither: the share of the way to the next colour at which a cell turns over
const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map((step) => (step + 0.5) / 16)

// The place in a list of colours for a value that runs along it: 0 is the
// first, 1 the second, and so on to `last`. Between two colours the cells
// of a small pattern turn over one by one, so a slope becomes a weave.
export function dithered(value: number, x: number, y: number, last: number): number {
  const held = Math.min(last, Math.max(0, value))
  const floor = Math.floor(held)
  return held - floor > BAYER[(y & 3) * 4 + (x & 3)] ? Math.min(last, floor + 1) : floor
}

// a number from 0 to 1 that is always the same for the same cell and seed
export function chance(x: number, y: number, seed = 0): number {
  const mixed = Math.sin(x * 127.1 + y * 311.7 + seed * 74.7) * 43758.5453
  return mixed - Math.floor(mixed)
}

const ease = (share: number) => share * share * (3 - 2 * share)

// a sheet of numbers from 0 to 1 that comes round again after 64 cells each way
const SHEET = 64
const LATTICE = Float32Array.from({ length: SHEET * SHEET }, (_, at) => chance(at % SHEET, Math.floor(at / SHEET), 11))
const knot = (x: number, y: number) => LATTICE[(y & (SHEET - 1)) * SHEET + (x & (SHEET - 1))]

// a smooth field from 0 to 1, for cloud and drifting light
export function drift(x: number, y: number, seed = 0): number {
  const [left, top] = [Math.floor(x), Math.floor(y)]
  const [across, down] = [ease(x - left), ease(y - top)]
  const [col, row] = [left + seed * 17, top + seed * 29]
  const upper = knot(col, row) + (knot(col + 1, row) - knot(col, row)) * across
  const lower = knot(col, row + 1) + (knot(col + 1, row + 1) - knot(col, row + 1)) * across
  return upper + (lower - upper) * down
}

export function put(frame: Frame, x: number, y: number, color: Rgb) {
  const at = (y * frame.cols + x) * 4
  frame.cells[at] = color[0]
  frame.cells[at + 1] = color[1]
  frame.cells[at + 2] = color[2]
  frame.cells[at + 3] = 255
}

// A scene: its colours by day and by night, and how it paints one frame.
// `ground` says which colours can stand under a row of the rail. `specks`
// says which colours are only ever single cells (a star, a glint, a fleck
// of sun); all the others can fill an area. `pace` is frames a second.
// `cell` is how many px a cell is: 4 when it is left out.
export type Scene = {
  colors: { light: readonly Rgb[]; dark: readonly Rgb[] }
  ground: { light: readonly number[]; dark: readonly number[] }
  specks: { light: readonly number[]; dark: readonly number[] }
  pace: number
  cell?: number
  paint: (frame: Frame, time: number, dark: boolean) => void
}
