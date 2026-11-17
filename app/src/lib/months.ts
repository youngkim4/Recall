// Calendar months as "YYYY-MM" text, which sorts and compares in date order.
// The arithmetic is done in UTC, so the reader's time zone never changes it.

export type MonthCount = { month: string; total: number }

const pad = (value: number) => String(value).padStart(2, '0')
const parts = (month: string) => ({ year: Number(month.slice(0, 4)), index: Number(month.slice(5, 7)) - 1 })

export function shiftMonth(month: string, by: number) {
  const { year, index } = parts(month)
  const moved = year * 12 + index + by
  return `${Math.floor(moved / 12)}-${pad((((moved % 12) + 12) % 12) + 1)}`
}

export function lastDayOf(month: string) {
  const { year, index } = parts(month)
  // day 0 of the next month is the last day of this one
  return `${month}-${pad(new Date(Date.UTC(year, index + 1, 0)).getUTCDate())}`
}

// One row a year, twelve places each; a place the months do not reach is null.
export function byYear(months: MonthCount[]) {
  if (!months.length) return []
  const known = new Map(months.map((entry) => [entry.month, entry]))
  const firstYear = parts(months[0].month).year
  const lastYear = parts(months[months.length - 1].month).year
  return Array.from({ length: lastYear - firstYear + 1 }, (_, offset) => {
    const year = String(firstYear + offset)
    return {
      year,
      slots: Array.from({ length: 12 }, (_slot, index) => known.get(`${year}-${pad(index + 1)}`) ?? null),
    }
  })
}

// The month with messages that lies `by` months from this one, or the nearest
// one further on in that direction; the month itself when there is none.
export function nextMonth(months: MonthCount[], month: string, by: number) {
  const wanted = shiftMonth(month, by)
  const candidates = months.filter((entry) => entry.total > 0 && (by > 0 ? entry.month >= wanted : entry.month <= wanted))
  if (!candidates.length) return month
  return (by > 0 ? candidates[0] : candidates[candidates.length - 1]).month
}
