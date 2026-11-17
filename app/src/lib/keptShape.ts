// The shape a list had the last time it was drawn: how many rows, and how
// tall. What stands in for the list while it loads takes the same room, so
// nothing on the page moves when the list arrives. Two numbers and no more:
// nothing of what the rows say is kept.

export type KeptShape = {
  rows: number
  // in pixels, to a tenth: a list of lines and bubbles is often half a pixel off a whole number
  height: number
}

const PREFIX = 'recall.shape.'
// more than this was never measured from a list on a page
const MOST_ROWS = 200
const MOST_HEIGHT = 20000

function isShape(value: unknown): value is KeptShape {
  if (!value || typeof value !== 'object') return false
  const { rows, height } = value as Partial<KeptShape>
  return (
    typeof rows === 'number' &&
    Number.isInteger(rows) &&
    rows >= 0 &&
    rows <= MOST_ROWS &&
    typeof height === 'number' &&
    Number.isFinite(height) &&
    height >= 0 &&
    height <= MOST_HEIGHT
  )
}

// the shape remembered under a name, or null when none is, or when it cannot be read
export function readShape(name: string): KeptShape | null {
  try {
    const stored = window.localStorage.getItem(PREFIX + name)
    const parsed: unknown = stored ? JSON.parse(stored) : null
    return isShape(parsed) ? { rows: parsed.rows, height: parsed.height } : null
  } catch {
    return null
  }
}

// Remembers a shape under a name. One that could not have been measured is
// left out, and one that is already remembered is not written again.
export function storeShape(name: string, shape: KeptShape) {
  const next = { rows: shape.rows, height: Math.round(shape.height * 10) / 10 }
  if (!isShape(next)) return
  const held = readShape(name)
  if (held && held.rows === next.rows && held.height === next.height) return
  try {
    window.localStorage.setItem(PREFIX + name, JSON.stringify(next))
  } catch {
    // remembering is optional: the list is drawn the same without it
  }
}
