// The moments a report picked out of a conversation: put in order, weighed,
// and placed along the span of days the report covers.

import type { AnalysisEvent } from '../types'
import { eventKind, plainQuote } from './reports.ts'

export type Moment = {
  key: string
  // "YYYY-MM-DD", or '' when the report gave no day that can be read
  day: string
  title: string
  detail: string
  // "Turning point"
  kind: string
  // the words it quotes, a line a message
  said: SaidLine[]
  // how much the moment matters, 0 to 1
  weight: number
}

// One quoted message. fromMe is null when the quote does not say who wrote
// it; sender is the name it gives for someone else, or ''.
export type SaidLine = { fromMe: boolean | null; sender: string; text: string }

export type SpanTick = { at: number; label: string }

const DAY_MS = 86_400_000
const MOST_MONTH_TICKS = 9
const MOST_YEAR_TICKS = 10
// a span longer than this is marked by its years
const MONTHS_BY_MONTH = 24

// the count of a day from a fixed start, or null when the text is not a real day
function dayNumber(day: string) {
  const found = /^(\d{4})-(\d{2})-(\d{2})/.exec(day)
  if (!found) return null
  const [year, month, date] = [Number(found[1]), Number(found[2]), Number(found[3])]
  const made = new Date(Date.UTC(year, month - 1, date))
  const real = made.getUTCFullYear() === year && made.getUTCMonth() === month - 1 && made.getUTCDate() === date
  return real ? made.getTime() / DAY_MS : null
}

const dayOf = (text: string) => (dayNumber(text) === null ? '' : text.slice(0, 10))

// The report was written from lines of the form "[2024-05-12 21:14] ME: words",
// and a quote often keeps the speaker, and sometimes the time, of each line.
const STAMP = /^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s*/
const VOICE = /^(ME|THEM|\?\?)\s*:\s*/
// a name is taken for a speaker only on a line that also carries its time
const NAMED = /^([^:]{1,40}?)\s*:\s+/
// a slash between two quoted pieces is where one message ended and the next began
const BETWEEN = /\n+|(?<=["”])\s+\/\s+(?=["“]|(?:ME|THEM)\s*:)/

function saidLine(line: string): SaidLine {
  const stamped = STAMP.test(line)
  const rest = line.replace(STAMP, '')
  const voice = VOICE.exec(rest)
  if (voice) {
    const fromMe = voice[1] === 'ME' ? true : voice[1] === 'THEM' ? false : null
    return { fromMe, sender: '', text: plainQuote(rest.slice(voice[0].length)) }
  }
  const named = stamped ? NAMED.exec(rest) : null
  if (named) return { fromMe: false, sender: named[1].trim(), text: plainQuote(rest.slice(named[0].length)) }
  return { fromMe: null, sender: '', text: plainQuote(rest) }
}

// What a moment quotes, as the messages it was: a line each, with who wrote
// it where the quote says.
export function readSaid(quote: string): SaidLine[] {
  return quote
    .split(BETWEEN)
    .map((line) => line.trim())
    .filter(Boolean)
    .map(saidLine)
    .filter((line) => line.text)
}

// Oldest first; a moment with no day goes last, in the order it was given.
export function readMoments(events: AnalysisEvent[]): Moment[] {
  return events
    .map((event, index) => ({
      key: `moment-${index}`,
      day: dayOf(String(event.date || '')),
      title: event.title || '',
      detail: event.detail || '',
      kind: eventKind(event.category),
      said: readSaid(event.quote || ''),
      weight: Math.max(0, Math.min(1, Number(event.score ?? 0) || 0)),
    }))
    .sort((a, b) => (a.day && b.day ? a.day.localeCompare(b.day) : Number(!a.day) - Number(!b.day)))
}

// The keys of the moments that matter most; of two that matter the same, the earlier.
export function leadingMoments(moments: Moment[], count: number) {
  const ranked = [...moments].sort((a, b) => b.weight - a.weight)
  return new Set(ranked.slice(0, Math.max(0, count)).map((moment) => moment.key))
}

// The moments under the year each one fell in, in order; '' for the ones with no day.
export function momentsByYear(moments: Moment[]) {
  return moments.reduce<Array<{ year: string; moments: Moment[] }>>((groups, moment) => {
    const year = moment.day.slice(0, 4)
    const open = groups[groups.length - 1]
    if (open && open.year === year) return [...groups.slice(0, -1), { year, moments: [...open.moments, moment] }]
    return [...groups, { year, moments: [moment] }]
  }, [])
}

// Where a day falls between the first day and the last, 0 to 1. A day outside
// is held at the nearer end; a span of a single day puts everything mid-way.
export function placeOnSpan(day: string, first: string, last: string) {
  const [at, from, to] = [dayNumber(day), dayNumber(first), dayNumber(last)]
  if (at === null || from === null || to === null) return null
  if (to <= from) return 0.5
  return Math.max(0, Math.min(1, (at - from) / (to - from)))
}

// the gap between marks that keeps their number at or under the most allowed
const stepFor = (count: number, most: number, steps: number[]) =>
  steps.find((step) => Math.ceil(count / step) <= most) ?? steps[steps.length - 1]

// The marks along a span of days: the first of each month when the span is
// short, the first of each year when it is long. A year is named where it
// starts, and marks are thinned until they have room.
export function spanTicks(first: string, last: string, locale?: string): SpanTick[] {
  const [from, to] = [dayNumber(first), dayNumber(last)]
  if (from === null || to === null || to <= from) return []
  const firstMonth = Number(first.slice(0, 4)) * 12 + Number(first.slice(5, 7)) - 1
  const lastMonth = Number(last.slice(0, 4)) * 12 + Number(last.slice(5, 7)) - 1
  const months = Array.from({ length: lastMonth - firstMonth }, (_, index) => firstMonth + index + 1)
  const byYear = months.length > MONTHS_BY_MONTH
  const starts = byYear ? months.filter((month) => month % 12 === 0) : months
  const step = byYear
    ? stepFor(starts.length, MOST_YEAR_TICKS, [1, 2, 5, 10, 20])
    : stepFor(starts.length, MOST_MONTH_TICKS, [1, 2, 3, 6])
  const name = new Intl.DateTimeFormat(locale, { month: 'short', timeZone: 'UTC' })
  return starts
    // counted from January, or from the year 0, so the same marks are kept however the span starts
    .filter((month) => (byYear ? Math.floor(month / 12) % step === 0 : (month % 12) % step === 0))
    .map((month) => {
      const year = Math.floor(month / 12)
      const start = Date.UTC(year, month % 12, 1)
      return {
        at: (start / DAY_MS - from) / (to - from),
        label: month % 12 === 0 ? String(year) : name.format(start),
      }
    })
    .filter((tick) => tick.at > 0 && tick.at < 1)
}
