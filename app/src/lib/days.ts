// Days as "YYYY-MM-DD" text, which sorts and compares in date order.
// All arithmetic and naming is done in UTC, so a change of clocks never moves
// a day and no time zone leaves one out.

import { lastDayOf, type MonthCount } from './months.ts'

// how many messages each day of a conversation holds, one count a day from the start
export type Daily = { start: string; counts: number[] }
// '' leaves that side open
export type DayRange = { since: string; until: string }

const DAY_MS = 86_400_000
const pad = (value: number) => String(value).padStart(2, '0')
const stamp = (day: string) => Date.UTC(Number(day.slice(0, 4)), Number(day.slice(5, 7)) - 1, Number(day.slice(8, 10)))
const dayOf = (ms: number) => new Date(ms).toISOString().slice(0, 10)
const dateOf = (day: string) => new Date(stamp(day))
// a day the calendar has: "2024-02-30" and "soon" are not days
const isDay = (day: string) => /^\d{4}-\d{2}-\d{2}$/.test(day) && dayOf(stamp(day)) === day

export function addDays(day: string, by: number) {
  return dayOf(stamp(day) + by * DAY_MS)
}

function daysBetween(from: string, to: string) {
  return Math.round((stamp(to) - stamp(from)) / DAY_MS)
}

// How many calendar days a stretch of time touches, both ends counted. The
// two are timestamps that begin with their day; 0 when either is missing.
export function daysSpanned(first?: string | null, last?: string | null) {
  if (!first || !last) return 0
  return daysBetween(first.slice(0, 10), last.slice(0, 10)) + 1
}

// The same day of another month, or the last day that month has.
export function addMonths(day: string, by: number) {
  const moved = Number(day.slice(0, 4)) * 12 + Number(day.slice(5, 7)) - 1 + by
  const month = `${Math.floor(moved / 12)}-${pad((((moved % 12) + 12) % 12) + 1)}`
  const last = lastDayOf(month)
  const wanted = `${month}-${day.slice(8, 10)}`
  return wanted > last ? last : wanted
}

export function lastDay(daily: Daily) {
  return daily.counts.length ? addDays(daily.start, daily.counts.length - 1) : ''
}

export function countOn(daily: Daily, day: string) {
  return daily.start ? (daily.counts[daysBetween(daily.start, day)] ?? 0) : 0
}

// Both ends are counted. An open side reaches the end of the conversation.
export function countBetween(daily: Daily, since: string, until: string) {
  if (!daily.counts.length) return 0
  const from = since ? Math.max(0, daysBetween(daily.start, since)) : 0
  const to = until ? Math.min(daily.counts.length - 1, daysBetween(daily.start, until)) : daily.counts.length - 1
  return daily.counts.slice(from, to + 1).reduce((sum, count) => sum + count, 0)
}

export function orderDays(a: string, b: string): DayRange {
  return a <= b ? { since: a, until: b } : { since: b, until: a }
}

// The part of a range that the conversation holds, with each side filled in.
// Null when no day of the conversation is in the range.
function closed(range: DayRange, daily: Daily): DayRange | null {
  const first = daily.start
  const last = lastDay(daily)
  const since = !range.since || range.since < first ? first : range.since
  const until = !range.until || range.until > last ? last : range.until
  return since <= until ? { since, until } : null
}

// A range that takes in the whole conversation says no more than no range.
export function tidyRange(range: DayRange, daily: Daily): DayRange {
  const whole = closed(range, daily)
  return whole && whole.since === daily.start && whole.until === lastDay(daily) ? { since: '', until: '' } : range
}

// The weeks of a month, each seven places from the day the week starts on
// (0 is Sunday). A place outside the month is null.
export function weeksOf(month: string, weekStart: number) {
  const first = `${month}-01`
  const length = Number(lastDayOf(month).slice(8, 10))
  const lead = (new Date(stamp(first)).getUTCDay() - weekStart + 7) % 7
  const places = [
    ...Array.from({ length: lead }, () => null),
    ...Array.from({ length }, (_, index) => `${month}-${pad(index + 1)}`),
  ]
  const trail = (7 - (places.length % 7)) % 7
  const padded = [...places, ...Array.from({ length: trail }, () => null)]
  return Array.from({ length: padded.length / 7 }, (_, week) => padded.slice(week * 7, week * 7 + 7))
}

// Every month from the first day to the last, with what it holds.
export function monthTotals(daily: Daily): MonthCount[] {
  const totals = new Map<string, number>()
  daily.counts.forEach((count, index) => {
    const month = addDays(daily.start, index).slice(0, 7)
    totals.set(month, (totals.get(month) ?? 0) + count)
  })
  return [...totals].map(([month, total]) => ({ month, total }))
}

// The last few months up to the last day, or null when the conversation is
// no longer than that.
export function lastMonthsRange(daily: Daily, count: number): DayRange | null {
  const until = lastDay(daily)
  if (!until) return null
  const since = addDays(addMonths(until, -count), 1)
  return since > daily.start ? { since, until } : null
}

// The part of a year the conversation covers.
export function yearRange(daily: Daily, year: string): DayRange | null {
  const last = lastDay(daily)
  if (!last || year < daily.start.slice(0, 4) || year > last.slice(0, 4)) return null
  return closed({ since: `${year}-01-01`, until: `${year}-12-31` }, daily)
}

const same = (a: DayRange | null, b: DayRange | null) => Boolean(a && b && a.since === b.since && a.until === b.until)

// A range as its days read. At least one side is a day, and two are in order.
function daysLabel(range: DayRange, locale?: string) {
  const format = new Intl.DateTimeFormat(locale, { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' })
  if (!range.until) return `From ${format.format(dateOf(range.since))}`
  if (!range.since) return `Through ${format.format(dateOf(range.until))}`
  if (range.since === range.until) return format.format(dateOf(range.since))
  // "Jun 10 – Jul 4, 2024": a shared year is written once
  return format.formatRange(dateOf(range.since), dateOf(range.until))
}

// What a range is called on the button that opens the calendar: the shortest
// name that is true of it. Every range has a name: a side that is not a day
// is read as open, and two days given back to front are read in order.
export function rangeLabel(range: DayRange, daily: Daily, locale?: string) {
  const since = isDay(range.since) ? range.since : ''
  const until = isDay(range.until) ? range.until : ''
  if ((!since && !until) || !daily.counts.length) return 'All dates'
  const asked = since && until ? orderDays(since, until) : { since, until }
  const whole = closed(asked, daily)
  // with no day of the conversation in it, it goes by its own days
  if (!whole) return daysLabel(asked, locale)
  if (whole.since === daily.start && whole.until === lastDay(daily)) return 'All dates'
  const year = whole.since.slice(0, 4)
  if (same(whole, yearRange(daily, year))) return year
  const month = whole.since.slice(0, 7)
  if (same(whole, closed({ since: `${month}-01`, until: lastDayOf(month) }, daily))) {
    const format = new Intl.DateTimeFormat(locale, { month: 'long', year: 'numeric', timeZone: 'UTC' })
    return format.format(dateOf(whole.since))
  }
  if (same(whole, lastMonthsRange(daily, 12))) return 'Last 12 months'
  if (same(whole, lastMonthsRange(daily, 3))) return 'Last 3 months'
  return daysLabel(whole, locale)
}
