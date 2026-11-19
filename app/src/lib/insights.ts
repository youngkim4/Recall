// What Insights asks the server to count (GET /api/stats, in
// recall_services/README.md): the filters of search, the words, and how to
// group the count. Nothing here talks to the server.

import { filterParams, type SearchFilters } from './searchFilters.ts'

export type StatsBy = 'day' | 'week' | 'month' | 'weekday' | 'hour' | 'weekday_hour' | 'person' | 'conversation'

export type StatsRow = {
  key: string
  label: string
  total: number
  sent: number
  received: number
  // on a row of a person or a conversation: the conversations it counts
  conversations?: string[]
}

export type StatsPayload = {
  by: string
  totals: { total: number; sent: number; received: number }
  // the first and last day of what was counted; null when nothing was
  first: string | null
  last: string | null
  rows: StatsRow[]
  // how many rows there were before the limit
  rowsTotal: number
}

// one count to ask for, in the words of the route
export type StatsAsk = Partial<Omit<SearchFilters, 'order'>> & {
  by: StatsBy
  query?: string
  limit?: number
}

// The count to ask for: only the filters that are set, the words when there
// are any, the grouping, and how many rows of people. The sort of search
// orders a list and means nothing to a count.
export function statsAsk(filters: SearchFilters, words: string, by: StatsBy, limit?: number): StatsAsk {
  // with the order at newest, filterParams leaves it out
  const set = filterParams({ ...filters, order: 'newest' }, false)
  const query = words.trim()
  return {
    by,
    ...(query ? { query } : {}),
    ...set,
    ...(limit ? { limit } : {}),
  }
}

// An ask is the key of a load (useLoad keys by a string): it travels as JSON.
export function readAsk(key: string): StatsAsk | null {
  try {
    const value: unknown = JSON.parse(key)
    return value && typeof value === 'object' && !Array.isArray(value) ? (value as StatsAsk) : null
  } catch {
    return null
  }
}

// The rows by weekday and hour ("0-09" is Monday, nine in the morning) as the
// heat map draws them: seven rows from Monday, twenty-four hours from midnight.
// A key that names no hour of the week is left out.
export function rhythmOf(rows: StatsRow[]): number[][] {
  const empty = Array.from({ length: 7 }, () => Array.from({ length: 24 }, () => 0))
  return rows.reduce((grid, row) => {
    const match = /^(\d)-(\d{2})$/.exec(row.key)
    const day = match ? Number(match[1]) : -1
    const hour = match ? Number(match[2]) : -1
    if (day < 0 || day > 6 || hour < 0 || hour > 23) return grid
    return grid.map((counts, at) =>
      at === day ? counts.map((count, h) => (h === hour ? count + row.total : count)) : counts,
    )
  }, empty)
}

// The key of one count for useLoad: the archive's path, a line break, and the
// ask as JSON. No path, no key: there is nothing to count.
export function statsKey(messagesPath: string, ask: StatsAsk) {
  return messagesPath ? `${messagesPath}\n${JSON.stringify(ask)}` : ''
}

// the path and the ask a key holds, or null for a key that holds neither
export function readStatsKey(key: string): { messagesPath: string; ask: StatsAsk } | null {
  const at = key.indexOf('\n')
  if (at < 1) return null
  const ask = readAsk(key.slice(at + 1))
  return ask ? { messagesPath: key.slice(0, at), ask } : null
}
