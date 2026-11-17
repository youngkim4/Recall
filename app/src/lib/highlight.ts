// Picking the searched-for words out of a message.

export type Segment = { text: string; match: boolean }

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
