// The size that sets one line across the room it has, as a newspaper sets its
// masthead: the width of a line grows with its size, so the size is scaled
// by the room over the width at a measured size. Never past `max`; under
// `min` the line is set at `min` and wraps. Nothing here touches the page.

type Fit = {
  // the room for the line, in pixels
  room: number
  // the width of the line at the size `at`, in pixels
  widthAt: number
  at: number
  min: number
  max: number
}

// a hair under the room, so rounding never pushes the last letter to a second line
const SLACK = 0.995

export function fitSize({ room, widthAt, at, min, max }: Fit): { size: number; wraps: boolean } {
  if (!(room > 0) || !(widthAt > 0) || !(at > 0)) return { size: max, wraps: false }
  const size = (at * room * SLACK) / widthAt
  if (size < min) return { size: min, wraps: true }
  return { size: Math.min(max, size), wraps: false }
}
