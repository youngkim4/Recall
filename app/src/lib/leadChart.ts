// The arithmetic of the chart beside the lead figure of the Overview.
import type { ActivityPoint } from './activity.ts'

export type YearTotal = { year: string; total: number }

// the messages of each year, oldest first, from months keyed 'YYYY-MM'
export function yearTotals(months: readonly ActivityPoint[]): YearTotal[] {
  const byYear = months.reduce<Map<string, number>>((years, month) => {
    const year = month.key.slice(0, 4)
    return new Map(years).set(year, (years.get(year) ?? 0) + month.total)
  }, new Map())
  return [...byYear.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([year, total]) => ({ year, total }))
}

// where the largest value is, so it can be named; -1 when there is none
export function peakAt(values: readonly number[]): number {
  return values.reduce((best, value, index) => (value > (values[best] ?? -Infinity) ? index : best), -1)
}
