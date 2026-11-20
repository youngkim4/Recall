// Who the archive's messages have been with lately, for the Overview: the
// last thirty days of the archive (not of the calendar: an archive can be
// behind today) against the thirty before them, with each person's last
// eight weeks for a small line. Nothing here talks to the server.

import { addDays } from './days.ts'
import type { StatsAsk, StatsRow } from './insights.ts'

export const LATELY_DAYS = 30
// weeks in a person's small line
const SPARK_WEEKS = 8

export type LatelyRow = StatsRow & {
  // how much more or less than the thirty days before, in whole percent; null with nothing then
  change: number | null
}

type Window = { since: string; until: string }

export function latelyWindows(lastDay: string): { now: Window; before: Window } {
  return {
    now: { since: addDays(lastDay, 1 - LATELY_DAYS), until: lastDay },
    before: { since: addDays(lastDay, 1 - 2 * LATELY_DAYS), until: addDays(lastDay, -LATELY_DAYS) },
  }
}

// The people of this window, the busiest first, each set against the window
// before. A person's row is keyed by their busiest thread, which can be
// another one from one window to the next, so rows are matched by any
// conversation they share.
export function latelyRows(now: StatsRow[], before: StatsRow[], count: number): LatelyRow[] {
  return now.slice(0, count).map((item) => {
    const ids = new Set(item.conversations ?? [item.key])
    const then = before.find((other) => (other.conversations ?? [other.key]).some((id) => ids.has(id)))
    const change = then && then.total > 0 ? Math.round(((item.total - then.total) / then.total) * 100) : null
    return { ...item, change }
  })
}

// what to ask for a person's small line: their last eight weeks, every conversation of theirs
export function sparkAsk(item: StatsRow, lastDay: string): StatsAsk {
  return {
    by: 'week',
    contacts: (item.conversations ?? [item.key]).join(','),
    since: addDays(lastDay, 1 - SPARK_WEEKS * 7),
    until: lastDay,
  }
}
