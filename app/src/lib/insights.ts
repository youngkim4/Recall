// What Insights asks the server to count (GET /api/stats, in
// recall_services/README.md): the filters of search, the words, and how to
// group the count. Nothing here talks to the server.

import { filterParams, type SearchFilters } from './searchFilters.ts'

export type StatsBy = 'day' | 'week' | 'month' | 'weekday' | 'hour' | 'weekday_hour' | 'person' | 'conversation'

// How the messages of a set are sent (recall_services/habits.py). A figure
// with nothing to count is null, never 0: a reply at once is 0 seconds.
export type Habits = {
  messages: number
  // the median seconds to the first reply of a talk
  youReply: number | null
  theyReply: number | null
  replies: { you: number; they: number }
  // the share of talks you started; a talk starts after 6 hours of quiet
  youStart: number | null
  talks: number
  streak: { days: number; first: string; last: string } | null
  // the share of messages from midnight to 5 a.m.
  lateNight: number | null
  // the mean characters of a text
  sentLength: number | null
  receivedLength: number | null
}

export type StatsRow = {
  key: string
  label: string
  total: number
  sent: number
  received: number
  // on a row of a person or a conversation: the conversations it counts
  conversations?: string[]
  // on a row of a person or a conversation, when habits were asked for
  habits?: Habits
}

export type StatsPayload = {
  by: string
  totals: { total: number; sent: number; received: number }
  // the first and last day of what was counted; null when nothing was
  first: string | null
  last: string | null
  // how many days hold a message of what was counted
  days: number
  rows: StatsRow[]
  // how many rows there were before the limit
  rowsTotal: number
  // when habits were asked for
  habits?: Habits
}

// one count to ask for, in the words of the route
export type StatsAsk = Partial<Omit<SearchFilters, 'order'>> & {
  by: StatsBy
  query?: string
  limit?: number
  // several conversations, comma separated, where `contact` names one
  contacts?: string
  // the habits of the set and of each row of people; they read every message of it
  habits?: boolean
}

// The count to ask for: only the filters that are set, the words when there
// are any, the grouping, and how many rows of people. The sort of search
// orders a list and means nothing to a count.
export function statsAsk(filters: SearchFilters, words: string, by: StatsBy, limit?: number, habits = false): StatsAsk {
  // with the order at newest, filterParams leaves it out
  const set = filterParams({ ...filters, order: 'newest' }, false)
  const query = words.trim()
  return {
    by,
    ...(query ? { query } : {}),
    ...set,
    ...(limit ? { limit } : {}),
    ...(habits ? { habits } : {}),
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
