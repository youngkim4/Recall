// The numbers of a chart on their way from one answer to the next. Pure
// numbers; useTweened.ts steps them on the screen.

// A series at another length: each new point is read off the old series at
// the same share of its length, between its two nearest points. So a line of
// 120 months can become one of 30 and still move as one line.
export function resample(values: readonly number[], length: number): number[] {
  if (length <= 0) return []
  if (!values.length) return Array.from({ length }, () => 0)
  if (values.length === 1 || length === 1) return Array.from({ length }, () => values[0])
  return Array.from({ length }, (_, index) => {
    const at = (index / (length - 1)) * (values.length - 1)
    const low = Math.floor(at)
    const high = Math.min(values.length - 1, low + 1)
    return values[low] + (values[high] - values[low]) * (at - low)
  })
}

// two series of one length, `share` of the way from the first to the second
export function mix(from: readonly number[], to: readonly number[], share: number): number[] {
  return to.map((value, index) => (from[index] ?? value) + (value - (from[index] ?? value)) * share)
}

// fast at first, settling at the end (cubic), for a share of the time from 0 to 1
export function easeOut(share: number): number {
  const time = Math.max(0, Math.min(1, share))
  return 1 - (1 - time) ** 3
}
