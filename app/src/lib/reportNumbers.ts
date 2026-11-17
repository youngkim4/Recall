// The figures of a report: the few it leads with, and what was shared and
// how messages were reacted to, by kind and by side.

import type { PreviewStats } from '../types'
import { daysSpanned } from './days.ts'
import { formatDate, formatDuration, formatNumber } from './format.ts'

export type KindCount = { label: string; count: number }

// a figure of a report: its name, the number, and one line about the number
export type KeyFigure = { label: string; value: string; note?: string }

type Counts = Record<string, number> | undefined

const SHARED: Array<[key: string, label: string]> = [
  ['photos', 'Photos'],
  ['videos', 'Videos'],
  ['gifs', 'GIFs'],
  ['audio', 'Audio'],
  ['documents', 'Files'],
  ['other', 'Other'],
]

// the names Messages gives its reactions
const REACTIONS: Array<[key: string, label: string]> = [
  ['loves', 'Loved'],
  ['likes', 'Liked'],
  ['laughs', 'Laughed'],
  ['emphasis', 'Emphasized'],
  ['questions', 'Questioned'],
  ['dislikes', 'Disliked'],
]

const whole = (value: unknown) => (typeof value === 'number' && Number.isFinite(value) ? Math.max(0, Math.round(value)) : 0)

// the kinds that have a count, the most first; equals keep the order they are named in
function kinds(counts: Counts, names: Array<[string, string]>): KindCount[] {
  return names
    .map(([key, label]) => ({ label, count: whole(counts?.[key]) }))
    .filter((kind) => kind.count > 0)
    .sort((a, b) => b.count - a.count)
}

export const sharedKinds = (attachments: Counts) => kinds(attachments, SHARED)
export const reactionKinds = (reactions: Counts) => kinds(reactions, REACTIONS)

// How a count splits between you and the other side, or null when the report does not say.
export function sidesOf(counts: Counts, what: 'attachments' | 'reactions') {
  const you = whole(counts?.[`sent_${what}`])
  const them = whole(counts?.[`received_${what}`])
  return you + them > 0 ? { you, them } : null
}

// The figures a report leads with. One the report does not have is left out.
export function keyFigures(stats: PreviewStats): KeyFigure[] {
  // days on the calendar: a late first message and an early last one still count their days
  const span = daysSpanned(stats.firstTimestamp, stats.lastTimestamp)
  const total = whole(stats.totalMessages)
  const active = whole(stats.activeDays)
  const gap = whole(stats.longestGapDays)
  const aDay = span ? Math.round(total / span) : 0
  const figures: Array<KeyFigure | null> = [
    total ? { label: 'Messages', value: formatNumber(total), ...(aDay ? { note: `${formatNumber(aDay)} a day` } : {}) } : null,
    active
      ? { label: 'Active days', value: formatNumber(active), ...(span > active ? { note: `of ${formatNumber(span)}` } : {}) }
      : null,
    stats.busiestDay && stats.busiestDayCount
      ? { label: 'Busiest day', value: formatNumber(stats.busiestDayCount), note: formatDate(stats.busiestDay) }
      : null,
    gap >= 1 ? { label: 'Longest gap', value: formatDuration(gap) } : null,
  ]
  return figures.filter((figure): figure is KeyFigure => figure !== null)
}
