// How many whole rows of one height a box has room for. A row that misses by
// less than a pixel counts: heights are measured in fractions of one. It is
// never fewer than `fewest`, whatever the box.
export function rowsThatFit(boxHeight: number, rowHeight: number, fewest: number) {
  if (!(rowHeight > 0) || !Number.isFinite(boxHeight)) return fewest
  return Math.max(fewest, Math.floor((boxHeight + 1) / rowHeight))
}
