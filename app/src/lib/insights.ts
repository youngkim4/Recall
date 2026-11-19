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

// What a grouping counts across: time (day, week, month), the turn of a week
// (weekday, hour), or who the messages are with (person, conversation).
export function groupKind(by: StatsBy): 'time' | 'cycle' | 'who' {
  if (by === 'person' || by === 'conversation') return 'who'
  if (by === 'weekday' || by === 'hour' || by === 'weekday_hour') return 'cycle'
  return 'time'
}

const FROM_WORDS: Record<string, string> = { me: 'from you', them: 'from others' }
const KIND_WORDS: Record<string, string> = { direct: 'in one-to-one chats', group: 'in group chats' }
const HAS_WORDS: Record<string, string> = {
  any: 'with an attachment',
  image: 'with a photo',
  video: 'with a video',
  audio: 'with audio',
  other: 'with a file',
  none: 'with no attachment',
}

type Question = {
  words: string
  filters: SearchFilters
  by: StatsBy
  // the name of the conversation the filters pick, where they pick one
  where?: string
  // the dates the filters pick, written out, where they pick any
  range?: string
}

// The question read back as one sentence, after the title Amplitude gives a
// chart: what is counted, then each filter that is set, then the grouping.
// "Messages with “love” from you in group chats, by month"
export function querySentence({ words, filters, by, where, range }: Question) {
  const word = words.trim()
  const parts = [
    word ? `Messages with “${word}”` : 'All messages',
    FROM_WORDS[filters.from],
    filters.contact && where ? `with ${where}` : KIND_WORDS[filters.kind],
    HAS_WORDS[filters.attachments],
  ].filter(Boolean)
  const grouped = `by ${by === 'weekday_hour' ? 'hour of the week' : by}`
  return [parts.join(' '), range, grouped].filter(Boolean).join(', ')
}

// The rows that hold the most, the most first; a tie keeps the order of the
// rows (time, for a time grouping). An empty row is never among them.
export function busiest(rows: StatsRow[], count: number): StatsRow[] {
  return rows
    .map((row, place) => ({ row, place }))
    .filter(({ row }) => row.total > 0)
    .sort((a, b) => b.row.total - a.row.total || a.place - b.place)
    .slice(0, count)
    .map(({ row }) => row)
}
