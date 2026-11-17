// What a share card draws, worked out from a conversation's own figures: the
// two voices over time, one day among the days around it, and the words for
// the ends of each. Days and months are read in UTC, the zone their keys are
// named in, so the reader's time zone never moves one.

import type { ActivityPoint, ActivityUnit } from './activity.ts'
import { addDays, countOn, lastDay, type Daily } from './days.ts'

// the two voices over a stretch of time, with what its first and last bars stand for
export type VoicesChart = {
  unit: ActivityUnit
  points: Array<{ sent: number; received: number }>
  first: string
  last: string
}

// some days in a row, with the place of the one the card is about
export type DaysChart = {
  counts: number[]
  at: number
  first: string
  last: string
  day: string
}

// a day is ranked among the days of its year only when there are this many
const DAYS_TO_RANK_AMONG = 28
const DAY = /^\d{4}-\d{2}-\d{2}$/

const dateOf = (key: string) =>
  new Date(Date.UTC(Number(key.slice(0, 4)), Number(key.slice(5, 7)) - 1, Number(key.slice(8, 10)) || 1))
const named = (key: string, parts: Intl.DateTimeFormatOptions, locale?: string) =>
  new Intl.DateTimeFormat(locale, { ...parts, timeZone: 'UTC' }).format(dateOf(key))
const shortMonth = (key: string, locale?: string) => named(key, { month: 'short', year: 'numeric' }, locale)
const shortDay = (key: string, locale?: string) => named(key, { month: 'short', day: 'numeric' }, locale)
const whole = (value: number) => (Number.isFinite(value) && value > 0 ? value : 0)

// The conversation's chart for a card, or null when there is nothing to draw.
export function voicesOf(
  activity: { unit: ActivityUnit; points: ActivityPoint[] } | null | undefined,
  locale?: string,
): VoicesChart | null {
  const points = (activity?.points ?? []).map((point) => ({ sent: whole(point.sent), received: whole(point.received) }))
  if (!activity || !points.some((point) => point.sent + point.received > 0)) return null
  const name = activity.unit === 'month' ? shortMonth : shortDay
  return {
    unit: activity.unit,
    points,
    first: name(activity.points[0].key, locale),
    last: name(activity.points[activity.points.length - 1].key, locale),
  }
}

// The day with up to `reach` days on each side of it, as far as the
// conversation goes. Null for a day the conversation does not hold.
export function daysAround(daily: Daily, day: string, reach = 15, locale?: string): DaysChart | null {
  const last = lastDay(daily)
  if (!daily.start || !DAY.test(day) || day < daily.start || day > last) return null
  const wantedFirst = addDays(day, -reach)
  const wantedLast = addDays(day, reach)
  const first = wantedFirst < daily.start ? daily.start : wantedFirst
  const end = wantedLast > last ? last : wantedLast
  const days = Array.from({ length: reach * 2 + 1 }, (_, index) => addDays(wantedFirst, index)).filter(
    (each) => each >= first && each <= end,
  )
  return {
    counts: days.map((each) => countOn(daily, each)),
    at: days.indexOf(day),
    first: shortDay(first, locale),
    last: shortDay(end, locale),
    day: shortDay(day, locale),
  }
}

// The share of the other days of its year, as far as the conversation goes,
// that held fewer messages than this day. Null for a day with nothing on it,
// and when the year has too few days to rank one among.
export function busierThan(daily: Daily, day: string): { share: number; year: string } | null {
  const last = lastDay(daily)
  if (!daily.start || !DAY.test(day) || day < daily.start || day > last) return null
  const held = countOn(daily, day)
  const year = day.slice(0, 4)
  const first = `${year}-01-01` < daily.start ? daily.start : `${year}-01-01`
  const end = `${year}-12-31` > last ? last : `${year}-12-31`
  const length = Math.round((dateOf(end).valueOf() - dateOf(first).valueOf()) / 86_400_000) + 1
  if (held <= 0 || length < DAYS_TO_RANK_AMONG) return null
  const others = Array.from({ length }, (_, index) => addDays(first, index)).filter((each) => each !== day)
  return { share: others.filter((each) => countOn(daily, each) < held).length / others.length, year }
}

// "Mar 2019 – Apr 2025" from the first and the last timestamp; '' when either is missing
export function spanOf(first?: string | null, last?: string | null, locale?: string) {
  if (!first || !last || !DAY.test(first.slice(0, 10)) || !DAY.test(last.slice(0, 10))) return ''
  const from = shortMonth(first.slice(0, 7), locale)
  const to = shortMonth(last.slice(0, 7), locale)
  return from === to ? from : `${from} – ${to}`
}

// "Saturday, November 16, 2024"; '' for what is not a day
export function longDay(day: string, locale?: string) {
  return DAY.test(day) ? named(day, { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' }, locale) : ''
}
