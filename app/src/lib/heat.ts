// The arithmetic behind the Overview's heat maps: a year of days, and the
// hours of the week.

import { addDays } from './days.ts'

// A count is drawn at one of five levels: 0 for nothing, 4 for the busiest
// quarter. A count is one level up for each step it is above.
export type HeatSteps = [number, number, number]

// The counts at the quarter marks of the days that have messages. Days with
// nothing take no part, or a quiet archive would put every busy day on top.
// A step never reaches the busiest count, so the days that share it take the
// top level even when they are more than a quarter of the days. When the busy
// days are all alike there are no steps: they are all as busy as it gets.
export function heatSteps(counts: number[]): HeatSteps {
  const busy = counts.filter((count) => count > 0).sort((a, b) => a - b)
  if (!busy.length) return [0, 0, 0]
  const most = busy[busy.length - 1]
  const nextMost = busy.findLast((count) => count < most) ?? 0
  const at = (share: number) => Math.min(busy[Math.ceil(busy.length * share) - 1], nextMost)
  return [at(0.25), at(0.5), at(0.75)]
}

export function heatLevel(count: number, steps: HeatSteps) {
  if (count <= 0) return 0
  return 1 + steps.filter((step) => count > step).length
}

const weekdayOf = (day: string) =>
  new Date(Date.UTC(Number(day.slice(0, 4)), Number(day.slice(5, 7)) - 1, Number(day.slice(8, 10)))).getUTCDay()

// The last `count` weeks up to the one that holds lastDay, oldest first. Each
// is seven places from the day the week starts on (0 is Sunday); the places
// after lastDay are null.
export function weeksEnding(lastDay: string, count: number, weekStart: number) {
  const intoWeek = (weekdayOf(lastDay) - weekStart + 7) % 7
  const firstDay = addDays(lastDay, -intoWeek - (count - 1) * 7)
  return Array.from({ length: count }, (_, week) =>
    Array.from({ length: 7 }, (_place, place) => {
      const day = addDays(firstDay, week * 7 + place)
      return day > lastDay ? null : day
    }),
  )
}

// The name of each month, over the first week that holds its first day.
export function monthMarks(weeks: Array<Array<string | null>>, locale?: string) {
  const short = new Intl.DateTimeFormat(locale, { month: 'short', timeZone: 'UTC' })
  return weeks.flatMap((days, week) => {
    const first = days.find((day) => day?.endsWith('-01'))
    if (!first) return []
    const month = new Date(Date.UTC(Number(first.slice(0, 4)), Number(first.slice(5, 7)) - 1, 1))
    return [{ week, text: short.format(month) }]
  })
}

// The hour of the week with the most messages. Rows run from Monday.
export function busiestHour(rhythm: number[][]) {
  const cells = rhythm.flatMap((hours, weekday) => hours.map((count, hour) => ({ weekday, hour, count })))
  const top = cells.reduce((best, cell) => (cell.count > best.count ? cell : best), { weekday: 0, hour: 0, count: 0 })
  return top.count > 0 ? top : null
}
