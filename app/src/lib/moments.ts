// What is worth opening today, as one list of moments: what was said on this
// day in earlier years, then the anniversaries that are near, then the
// conversations that have gone quiet. Each moment is a thing of its own, and
// a page shows it as one.
import type { MemoriesPayload } from '../types'
import { formatDate, plural } from './format.ts'

export type MomentKind = 'today' | 'anniversary' | 'quiet'

export type Moment = {
  key: string
  kind: MomentKind
  chatId: string
  // the saved name as the server sent it, or ''
  savedName: string
  // how this moment relates to today: "2 years ago today", "In 6 days", "Quiet since Oct 25, 2018"
  when: string
  // fromMe is null when the export does not say who sent the message
  quote?: { text: string; fromMe: boolean | null }
  // in a group, who wrote the quoted message
  quotedFrom?: string
  line: string
  // the day the moment is of, when its quoted message was sent, and that message's id
  day?: string
  at?: string | null
  quotedId?: string
}

// the most of each kind that a list shows, unless it is told otherwise
const MOST: Record<MomentKind, number> = { today: 3, anniversary: 2, quiet: 2 }
// The Overview's shelf: every moment of this day in earlier years that the
// server sends (six at most), then every anniversary near (four at most),
// three cards a row. The quiet are a list of their own beside Lately.
const SHELF: Record<MomentKind, number> = { today: Infinity, anniversary: Infinity, quiet: 0 }

function yearsAgoToday(years: number) {
  return years === 1 ? 'A year ago today' : `${years} years ago today`
}

function daysAway(days: number) {
  if (days === 0) return 'Today'
  return days === 1 ? 'Tomorrow' : `In ${days} days`
}

export function momentsOf(memories: MemoriesPayload, most: Record<MomentKind, number> = MOST): Moment[] {
  const today = memories.onThisDay.slice(0, most.today).map(
    (item): Moment => ({
      key: `day-${item.chatId}-${item.year}`,
      kind: 'today',
      chatId: item.chatId,
      savedName: item.name,
      when: yearsAgoToday(item.yearsAgo),
      quote: item.preview ? { text: item.preview, fromMe: item.previewFromMe } : undefined,
      quotedFrom: item.previewSender || undefined,
      line: `${plural(item.count, 'message')} that day`,
      day: item.day,
      at: item.previewAt,
      quotedId: item.previewId || undefined,
    }),
  )
  const anniversaries = memories.anniversaries.slice(0, most.anniversary).map(
    (item): Moment => ({
      key: `anniversary-${item.chatId}`,
      kind: 'anniversary',
      chatId: item.chatId,
      savedName: item.name,
      when: daysAway(item.inDays),
      // the first message, as the line says, so the card holds more than a count of years
      quote: item.preview ? { text: item.preview, fromMe: item.previewFromMe ?? null } : undefined,
      quotedFrom: item.previewSender || undefined,
      line: `${plural(item.years, 'year')} since the first message`,
      // the day of the quoted message (the first that holds words can come a
      // day after a first attachment); with no quote, the day it began
      day: (item.previewAt || item.date).slice(0, 10),
      at: item.previewAt ?? null,
      quotedId: item.previewId || undefined,
    }),
  )
  // "quiet" is measured to the end of the archive, so the date is given rather
  // than a length of time counted to today
  const quiet = memories.reconnect.slice(0, most.quiet).map(
    (item): Moment => ({
      key: `quiet-${item.chatId}`,
      kind: 'quiet',
      chatId: item.chatId,
      savedName: item.name,
      when: `Quiet since ${formatDate(item.lastDate)}`,
      line: plural(item.count, 'message'),
      // the last day anything was said
      day: item.lastDate.slice(0, 10),
    }),
  )
  return [...today, ...anniversaries, ...quiet]
}

// the cards of the Overview's shelf of moments
export const shelfOf = (memories: MemoriesPayload) => momentsOf(memories, SHELF)

// the conversations that have gone quiet, as many as are asked for
export const quietOf = (memories: MemoriesPayload, count: number) =>
  momentsOf(memories, { today: 0, anniversary: 0, quiet: count })
