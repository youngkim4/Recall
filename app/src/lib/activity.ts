// Messages over time, as the server counts them: by day for a short stretch,
// by month for a long one. A day's key is "YYYY-MM-DD", a month's "YYYY-MM".

export type ActivityUnit = 'day' | 'month'
// what one point of a chart counts; a week (Insights) is keyed by its Monday
export type ChartUnit = ActivityUnit | 'week'
export type ActivityPoint = { key: string; total: number; sent: number; received: number }

const ROUND_STEPS = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
// more day labels than this crowd the axis
const DAY_LABELS = 8
const EACH_DAY_UP_TO = 10
const EACH_MONTH_UP_TO = 12

// the next round number at or above value, close enough that the tallest bar nearly fills its side
export function roundUp(value: number) {
  if (value <= 0) return 0
  const power = 10 ** Math.floor(Math.log10(value))
  const step = ROUND_STEPS.find((multiple) => multiple * power >= value) ?? 10
  return Math.round(step * power)
}

// A key's day in UTC, which is also the zone it is named in, so the reader's
// time zone never changes it.
const dateOf = (key: string) =>
  new Date(Date.UTC(Number(key.slice(0, 4)), Number(key.slice(5, 7)) - 1, Number(key.slice(8, 10)) || 1))

// "March 2024" for a month, "Sun, Dec 8, 2024" for a day, "Week of Mar 3, 2025" for a week
export function pointName(key: string, unit: ChartUnit, locale?: string) {
  if (unit === 'week') {
    const monday = new Intl.DateTimeFormat(locale, { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' })
    return `Week of ${monday.format(dateOf(key))}`
  }
  const parts: Intl.DateTimeFormatOptions =
    unit === 'month'
      ? { month: 'long', year: 'numeric' }
      : { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' }
  return new Intl.DateTimeFormat(locale, { ...parts, timeZone: 'UTC' }).format(dateOf(key))
}

// index is the point a label stands under; atEdge starts it on the chart's edge instead
export type AxisLabel = { index: number; text: string; atEdge?: boolean }

// Every month is named when there are few. Otherwise the opening year and
// each January are, by year.
function monthLabels(points: ActivityPoint[], locale?: string): AxisLabel[] {
  if (points.length <= EACH_MONTH_UP_TO) {
    const short = new Intl.DateTimeFormat(locale, { month: 'short', timeZone: 'UTC' })
    return points.map((point, index) => ({ index, text: short.format(dateOf(point.key)) }))
  }
  const januaries = points
    .map((point, index) => ({ index, text: point.key.slice(0, 4), month: point.key.slice(5, 7) }))
    .filter((entry) => entry.month === '01')
    .map(({ index, text }) => ({ index, text }))
  // The opening year is named too, unless its label would crowd the next one.
  // Its month is too narrow to center a year under, so it starts on the edge.
  const roomForFirst = !januaries.length || januaries[0].index >= Math.max(3, points.length / 12)
  return roomForFirst ? [{ index: 0, text: points[0].key.slice(0, 4), atEdge: true }, ...januaries] : januaries
}

// Days are numbered at even steps. The first carries its month, and so does
// the first label of each month after it.
function dayLabels(points: ActivityPoint[], locale?: string): AxisLabel[] {
  const withMonth = new Intl.DateTimeFormat(locale, { month: 'short', day: 'numeric', timeZone: 'UTC' })
  const each = points.length <= EACH_DAY_UP_TO
  const step = each ? 1 : Math.ceil(points.length / DAY_LABELS)
  const shown = points.map((point, index) => ({ point, index })).filter(({ index }) => index % step === 0)
  return shown.map(({ point, index }, place) => {
    const newMonth = place === 0 || shown[place - 1].point.key.slice(0, 7) !== point.key.slice(0, 7)
    return {
      index,
      text: newMonth ? withMonth.format(dateOf(point.key)) : String(Number(point.key.slice(8, 10))),
      // a narrow first day cannot hold its month centered under it
      ...(place === 0 && !each ? { atEdge: true } : {}),
    }
  })
}

// weeks are keyed by their Mondays, and labelled as days are
export function axisLabels(points: ActivityPoint[], unit: ChartUnit, locale?: string): AxisLabel[] {
  return unit === 'month' ? monthLabels(points, locale) : dayLabels(points, locale)
}
