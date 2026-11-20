// The habits of Insights read for the screen: a wait in the unit that reads
// well, the months of a streak, the busiest hour of the week, and the week as
// a grid of hours. Nothing here talks to the server.

import type { StatsRow } from './insights.ts'

const MISSING = '—'
const MINUTE = 60
const HOUR = 3600
// under this many hours a wait keeps one decimal: "1.5 hr"
const TENTHS_UNDER_HOURS = 10
const HOURS_PER_DAY = 24

const unit = (name: 'second' | 'minute' | 'hour', digits = 0) =>
  new Intl.NumberFormat(undefined, { style: 'unit', unit: name, unitDisplay: 'short', maximumFractionDigits: digits })
const seconds = unit('second')
const minutes = unit('minute')
const hoursTenths = unit('hour', 1)
const hours = unit('hour')
const monthYear = new Intl.DateTimeFormat(undefined, { month: 'short', year: 'numeric' })

// "42 sec", "16 min", "1.5 hr", "36 hr"
export function formatWait(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) return MISSING
  if (value < MINUTE) return seconds.format(Math.round(value))
  if (value < HOUR) return minutes.format(Math.round(value / MINUTE))
  const inHours = value / HOUR
  return inHours < TENTHS_UNDER_HOURS ? hoursTenths.format(inHours) : hours.format(Math.round(inHours))
}

// the mean characters of a text, whole
export function formatLength(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) return MISSING
  return String(Math.round(value))
}

// a local day "YYYY-MM-DD" as a date at noon, so no zone moves it to the day before
function dayOf(key: string) {
  const [year, month, day] = key.split('-').map(Number)
  return new Date(year, month - 1, day, 12)
}

// "Jul 2022 to Aug 2025"; one month is named once
export function formatSpan(first: string, last: string) {
  const from = monthYear.format(dayOf(first))
  const to = monthYear.format(dayOf(last))
  return from === to ? from : `${from} to ${to}`
}

// the server counts weekdays from Monday ("0"); Date counts them from Sunday
const dateWeekday = (serverWeekday: number) => (serverWeekday + 1) % 7

function readCell(key: string) {
  const [weekday, hour] = key.split('-').map(Number)
  return { weekday, hour }
}

// The hour of the week that holds the most messages, the first of a tie, with
// its weekday as the server counts it (Monday is 0). Null with no message.
export function busiestHour(rows: StatsRow[]) {
  const top = rows.reduce<StatsRow | null>((best, row) => (row.total > (best?.total ?? 0) ? row : best), null)
  return top ? { ...readCell(top.key), total: top.total } : null
}

export type WeekCell = { hour: number; total: number; level: number }
export type WeekRow = { weekday: number; cells: WeekCell[] }

// The week as rows of 24 hours, in the reader's order of days (`order` holds
// the weekdays as Date counts them, 0 is Sunday). A cell's level is its count
// against the busiest hour, from 0 to 1.
export function weekGrid(rows: StatsRow[], order: number[]): WeekRow[] {
  const totals = new Map<string, number>()
  for (const row of rows) {
    const { weekday, hour } = readCell(row.key)
    totals.set(`${dateWeekday(weekday)}-${hour}`, row.total)
  }
  const most = Math.max(0, ...totals.values())
  return order.map((weekday) => ({
    weekday,
    cells: Array.from({ length: HOURS_PER_DAY }, (_, hour) => {
      const total = totals.get(`${weekday}-${hour}`) ?? 0
      return { hour, total, level: most ? total / most : 0 }
    }),
  }))
}
