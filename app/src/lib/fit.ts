// the most a row gives up of its own height so that one more row is shown
const SQUEEZE = 0.05

// How many rows a box shows when its rows share its height equally. It is
// the count that leaves a row nearest its own height: one more row is shown
// when each row is then a twentieth shorter at most. So the rows of a list
// that is fitted to a box are never plainly taller than the rows of a list
// beside it. It is never fewer than `fewest`, whatever the box.
export function rowsThatFit(boxHeight: number, rowHeight: number, fewest: number) {
  if (!(rowHeight > 0) || !Number.isFinite(boxHeight)) return fewest
  const whole = Math.floor(boxHeight / rowHeight)
  const oneMore = boxHeight / (whole + 1) >= rowHeight * (1 - SQUEEZE)
  return Math.max(fewest, oneMore ? whole + 1 : whole)
}
