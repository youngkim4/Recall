// The arithmetic of the line under the third figure of the Overview: how
// much of the archive its busiest conversations hold.

const round = (value: number) => Math.round(value * 100) / 100

// The largest parts as shares of the whole, largest first, at most `count`
// of them. The parts never come to more than the whole.
export function topShares(parts: number[], whole: number, count: number) {
  if (whole <= 0) return []
  return [...parts]
    .sort((a, b) => b - a)
    .slice(0, count)
    .reduce<number[]>((shares, part) => {
      const left = 1 - shares.reduce((sum, share) => sum + share, 0)
      return [...shares, round(Math.max(0, Math.min(left, part / whole)))]
    }, [])
}
