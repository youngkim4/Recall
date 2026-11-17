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

// the most of each kind that a page shows
const MOST: Record<MomentKind, number> = { today: 3, anniversary: 2, quiet: 2 }

function yearsAgoToday(years: number) {
  return years === 1 ? 'A year ago today' : `${years} years ago today`
}

function daysAway(days: number) {
  if (days === 0) return 'Today'
  return days === 1 ? 'Tomorrow' : `In ${days} days`
}

export function momentsOf(memories: MemoriesPayload): Moment[] {
  const today = memories.onThisDay.slice(0, MOST.today).map(
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
  const anniversaries = memories.anniversaries.slice(0, MOST.anniversary).map(
    (item): Moment => ({
      key: `anniversary-${item.chatId}`,
      kind: 'anniversary',
      chatId: item.chatId,
      savedName: item.name,
      when: daysAway(item.inDays),
      line: `${plural(item.years, 'year')} since the first message`,
      // the day it began
      day: item.date.slice(0, 10),
    }),
  )
  // "quiet" is measured to the end of the archive, so the date is given rather
  // than a length of time counted to today
  const quiet = memories.reconnect.slice(0, MOST.quiet).map(
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
