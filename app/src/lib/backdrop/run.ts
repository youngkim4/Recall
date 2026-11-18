// Keeps a scene painted on a canvas: at the scene's pace while the page is
// seen, one still picture where motion is not wanted, and again whenever
// the room it is painted for changes.
import { FEATHER, frameOf, type Frame, type Scene } from './craft.ts'

// a cell is 4px unless the scene says otherwise
const CELL = 4
// the still picture is a moment a little way into the scene
export const STILL_AT = 6

// the room a scene is painted for, in px: the window, and where the rail's rows stand
export type Room = { width: number; height: number; railRight: number; navBottom: number; footTop: number }

// the frame for a room: the rows of the rail keep a cell of clear ground around them
export function frameFor(room: Room, cell = CELL): Frame {
  const [cols, rows] = [Math.max(1, Math.ceil(room.width / cell)), Math.max(1, Math.ceil(room.height / cell))]
  const railCols = Math.min(cols, Math.ceil(room.railRight / cell))
  const navRows = Math.min(rows, Math.ceil(room.navBottom / cell) + 1)
  const footRow = Math.max(navRows, Math.min(rows, Math.floor(room.footTop / cell) - 1))
  // the slope from a row to the open picture is as long in px whatever the size of a cell
  return frameOf(cols, rows, railCols, navRows, footRow, Math.round((FEATHER * CELL) / cell))
}

// two rooms that give the same frame
export function sameFrame(a: Room, b: Room, cell = CELL): boolean {
  const [x, y] = [frameFor(a, cell), frameFor(b, cell)]
  return x.cols === y.cols && x.rows === y.rows && x.railCols === y.railCols && x.navRows === y.navRows && x.footRow === y.footRow
}

export type Running = { stop: () => void; refit: () => void }

export function runScene(canvas: HTMLCanvasElement, scene: Scene, dark: boolean, roomNow: () => Room): Running {
  const context = canvas.getContext('2d')
  if (!context) return { stop: () => {}, refit: () => {} }
  const pen = context
  const cell = scene.cell ?? CELL
  const still = window.matchMedia('(prefers-reduced-motion: reduce)')
  let room: Room | null = null
  let frame: Frame | null = null
  let image: ImageData | null = null
  let asked = 0

  // nothing is painted where it cannot be seen: a hidden page, or a page laid out as paper
  const unseen = () => document.hidden || document.documentElement.hasAttribute('data-paper')

  function paint() {
    if (!frame || !image || unseen()) return
    scene.paint(frame, still.matches ? STILL_AT : performance.now() / 1000, dark)
    pen.putImageData(image, 0, 0)
  }

  function fit() {
    asked = 0
    const next = roomNow()
    if (room && sameFrame(room, next, cell) && (frame || next.width < 1)) return
    room = next
    // with no rail there is nothing to paint
    if (next.width < 1) {
      frame = null
      image = null
      canvas.width = 0
      canvas.style.width = '0px'
      return
    }
    frame = frameFor(next, cell)
    image = new ImageData(frame.cells, frame.cols, frame.rows)
    canvas.width = frame.cols
    canvas.height = frame.rows
    // the last cell is cut to the rail: the picture never reaches past it
    canvas.style.width = `${next.width}px`
    canvas.style.height = `${next.height}px`
    paint()
  }

  // a room that changes many times in a frame (the rail folding) is fitted once a frame
  function refit() {
    if (!asked) asked = window.requestAnimationFrame(fit)
  }

  fit()
  const timer = window.setInterval(() => {
    if (!still.matches) paint()
  }, 1000 / scene.pace)
  window.addEventListener('resize', refit)
  document.addEventListener('visibilitychange', paint)
  still.addEventListener('change', paint)
  return {
    refit,
    stop: () => {
      window.clearInterval(timer)
      window.cancelAnimationFrame(asked)
      window.removeEventListener('resize', refit)
      document.removeEventListener('visibilitychange', paint)
      still.removeEventListener('change', paint)
    },
  }
}
