import type { KeyboardEvent } from 'react'

const MOVES: Partial<Record<string, (index: number, count: number) => number>> = {
  ArrowDown: (index, count) => Math.min(count - 1, index + 1),
  ArrowUp: (index) => Math.max(0, index - 1),
  Home: () => 0,
  End: (_index, count) => count - 1,
}

// Where a key takes the highlight in a list of count items, or null when the
// key does not move it. The ends hold: a list does not wrap around.
export function stepIndex(key: string, index: number, count: number): number | null {
  const move = MOVES[key]
  return move && count > 0 ? move(index, count) : null
}

// The arrow keys move focus through a list, so the list is one stop for Tab
// however long it is. Put this on the list's keydown, mark each item with
// data-roving, and give the current item tabIndex 0 and the others -1.
export function rovingKeyDown(event: KeyboardEvent<HTMLElement>) {
  const items = [...event.currentTarget.querySelectorAll<HTMLElement>('[data-roving]')]
  const index = items.indexOf(document.activeElement as HTMLElement)
  if (index === -1) return
  const next = stepIndex(event.key, index, items.length)
  if (next === null) return
  event.preventDefault()
  items[next]?.focus()
}
