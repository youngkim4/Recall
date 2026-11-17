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

// The one day of a month that Tab stops on: the first of the wanted days that
// the month holds, kept inside the conversation. With none of them it is the
// month's first day that the conversation holds, or in the month the
// conversation ends in, its last day. A month always has such a day.
export function tabStopIn(month: string, wanted: Array<string | null | undefined>, first: string, last: string) {
  const held = (day: string) => (day < first ? first : day > last ? last : day)
  const inMonth = wanted.find((day): day is string => Boolean(day?.startsWith(month)))
  if (inMonth) return held(inMonth)
  return month === last.slice(0, 7) ? last : held(`${month}-01`)
}

type Span = { top: number; height: number }

// How far down a list has to be scrolled for one of its rows to be in view.
// `view` is what the list shows now and `row` is where the row lies in the
// whole list. 'nearest' moves no further than it must, and not at all for a
// row in view; 'center' puts the row in the middle.
export function scrollToShow(view: Span, row: Span, how: 'nearest' | 'center') {
  if (how === 'center') return Math.max(0, row.top - (view.height - row.height) / 2)
  if (row.top < view.top) return row.top
  const end = row.top + row.height
  return end > view.top + view.height ? end - view.height : view.top
}
