// A ramp of colours, and the colour at one place along it. A control that
// shows a level by its colour reads both from the same ramp.

type Rgb = [number, number, number]

const HEX = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i

function rgbOf(color: string): Rgb {
  const parts = HEX.exec(color)
  return parts ? [parseInt(parts[1], 16), parseInt(parts[2], 16), parseInt(parts[3], 16)] : [0, 0, 0]
}

// The colour `along` the ramp, 0 at its first colour and 1 at its last,
// mixed evenly between the two colours it stands between.
export function toneAt(ramp: string[], along: number): string {
  if (!ramp.length) return 'transparent'
  const place = Math.min(1, Math.max(0, Number.isFinite(along) ? along : 0)) * (ramp.length - 1)
  const from = Math.min(ramp.length - 1, Math.floor(place))
  const to = Math.min(ramp.length - 1, from + 1)
  const share = place - from
  const [a, b] = [rgbOf(ramp[from]), rgbOf(ramp[to])]
  const mixed = a.map((part, index) => Math.round(part + (b[index] - part) * share))
  return `rgb(${mixed.join(' ')})`
}

// The ramp as a background, its colours from left to right.
export function rampCss(ramp: string[]): string {
  const colors = ramp.length > 1 ? ramp : [ramp[0], ramp[0]]
  return `linear-gradient(90deg, ${colors.join(', ')})`
}
