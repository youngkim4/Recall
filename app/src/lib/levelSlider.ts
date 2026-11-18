// The arithmetic of a slider whose values are a few named steps.

type Span = { left: number; width: number }

// How far along the track a point is, from 0 at the first step to 1 at the
// last. The first and last steps stand an inset in from the ends, where the
// middle of the knob can reach.
export function alongAt(x: number, track: Span, inset: number): number {
  const reach = track.width - inset * 2
  if (reach <= 0) return 0
  return Math.min(1, Math.max(0, (x - track.left - inset) / reach))
}

// How far past the middle between two steps a held knob must go before its
// step changes, as a share of the way from one step to the next.
const HOLD = 0.15

// The step of a knob that a pointer holds `along` the track, when it is on
// step `current`. The step changes only when the pointer is well past half
// way to another step. So a hand that rests between two steps does not flip
// from one to the other, and a step that was just taken is not given back
// by a small move the other way.
export function heldStep(along: number, count: number, current: number): number {
  if (count < 2) return 0
  const at = Math.min(1, Math.max(0, along)) * (count - 1)
  return Math.abs(at - current) <= 0.5 + HOLD ? current : Math.round(at)
}

// The step nearest a point on the track.
export function stepAt(x: number, track: Span, inset: number, count: number): number {
  if (count <= 1) return 0
  return Math.round(alongAt(x, track, inset) * (count - 1))
}

const MOVES: Partial<Record<string, (index: number, count: number) => number>> = {
  ArrowRight: (index, count) => Math.min(count - 1, index + 1),
  ArrowUp: (index, count) => Math.min(count - 1, index + 1),
  ArrowLeft: (index) => Math.max(0, index - 1),
  ArrowDown: (index) => Math.max(0, index - 1),
  Home: () => 0,
  End: (_index, count) => count - 1,
}

// Where a key takes the knob, or null when the key does not move it. The ends hold.
export function stepByKey(key: string, index: number, count: number): number | null {
  const move = MOVES[key]
  return move && count > 0 ? move(index, count) : null
}
