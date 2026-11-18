// Picking the searched-for words out of a message.

export type Segment = { text: string; match: boolean }

// The text in pieces, by the ranges the server's index matched: [start, end)
// in UTF-16 units, as JavaScript counts. Ranges that are not numbers, are
// empty or fall outside the text are left out; ranges that overlap are joined.
export function splitByRanges(text: string, ranges: ReadonlyArray<readonly [number, number]>): Segment[] {
  if (!text) return []
  const whole = (value: unknown) => Number.isInteger(value)
  const clipped = ranges
    .filter((range) => Array.isArray(range) && whole(range[0]) && whole(range[1]))
    .map(([start, end]) => [Math.max(0, start), Math.min(text.length, end)] as const)
    .filter(([start, end]) => end > start)
    .sort((a, b) => a[0] - b[0])
  const joined = clipped.reduce<Array<readonly [number, number]>>((all, range) => {
    const last = all.at(-1)
    if (last && range[0] <= last[1]) return [...all.slice(0, -1), [last[0], Math.max(last[1], range[1])]]
    return [...all, range]
  }, [])
  const pieces = joined.flatMap(([start, end], at) => {
    const from = at === 0 ? 0 : joined[at - 1][1]
    return [
      { text: text.slice(from, start), match: false },
      { text: text.slice(start, end), match: true },
    ]
  })
  const rest = { text: text.slice(joined.at(-1)?.[1] ?? 0), match: false }
  return [...pieces, rest].filter((segment) => segment.text)
}

// The text in pieces, in order: the stretches that match the term, whatever
// their case, and the stretches between them.
export function splitByTerm(text: string, term: string): Segment[] {
  if (!text) return []
  const needle = term.trim()
  if (!needle) return [{ text, match: false }]
  // the term is matched as written, not as a pattern
  const literal = needle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  return text
    .split(new RegExp(`(${literal})`, 'gi'))
    .map((part, index) => ({ text: part, match: index % 2 === 1 }))
    .filter((segment) => segment.text)
}
