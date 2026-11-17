// Numbers and dates follow the reader's locale; the words around them are
// English, like the rest of the app.

const numberFormat = new Intl.NumberFormat()
const moneyFormat = new Intl.NumberFormat(undefined, {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 2,
})
const dateFormat = new Intl.DateTimeFormat(undefined, {
  month: 'short',
  day: 'numeric',
  year: 'numeric',
})
const dayFormat = new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric' })
const longDateFormat = new Intl.DateTimeFormat(undefined, {
  month: 'long',
  day: 'numeric',
  year: 'numeric',
})
const timeFormat = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' })
const longMonthFormat = new Intl.DateTimeFormat(undefined, { month: 'long', year: 'numeric' })

const MISSING = '—'

export function formatNumber(value: number | null | undefined) {
  if (!Number.isFinite(value)) return MISSING
  return numberFormat.format(Number(value))
}

export function formatMoney(value: number | null | undefined) {
  if (!Number.isFinite(value)) return MISSING
  return moneyFormat.format(Number(value))
}

export function formatPercent(fraction: number | null | undefined) {
  if (!Number.isFinite(fraction)) return MISSING
  return `${Math.round(Number(fraction) * 100)}%`
}

// The archive stores local wall-clock times with no zone. Date-only strings
// would parse as UTC midnight and can land on the previous day, so those are
// built from their parts instead.
export function parseDate(value?: string | number | Date | null): Date | null {
  if (value === null || value === undefined || value === '') return null
  if (value instanceof Date) return Number.isNaN(value.valueOf()) ? null : value
  if (typeof value === 'number') return new Date(value)
  const dateOnly = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value.trim())
  const date = dateOnly
    ? new Date(Number(dateOnly[1]), Number(dateOnly[2]) - 1, Number(dateOnly[3]))
    : new Date(value.trim().replace(' ', 'T'))
  return Number.isNaN(date.valueOf()) ? null : date
}

// "Mar 23, 2024"
export function formatDate(value?: string | number | Date | null) {
  const date = parseDate(value)
  return date ? dateFormat.format(date) : MISSING
}

// "March 23, 2024"
export function formatLongDate(value?: string | number | Date | null) {
  const date = parseDate(value)
  return date ? longDateFormat.format(date) : MISSING
}

// "Mar 23 – Dec 17, 2024"; the shared year is written once
export function formatDateRange(
  start?: string | number | Date | null,
  end?: string | number | Date | null,
) {
  const from = parseDate(start)
  const to = parseDate(end)
  if (!from && !to) return MISSING
  if (!from || !to) return dateFormat.format((from || to) as Date)
  return dateFormat.formatRange(from, to)
}

// "May 21, 3:04 PM" this year, "May 21, 2025, 3:04 PM" otherwise
export function formatDateTime(value?: string | number | Date | null) {
  const date = parseDate(value)
  if (!date) return MISSING
  const day =
    date.getFullYear() === new Date().getFullYear() ? dayFormat.format(date) : dateFormat.format(date)
  return `${day}, ${timeFormat.format(date)}`
}

// "2024-03" -> "March 2024"
export function formatLongMonth(value: string) {
  const match = /^(\d{4})-(\d{2})/.exec(value)
  if (!match) return value
  return longMonthFormat.format(new Date(Number(match[1]), Number(match[2]) - 1, 1))
}

// recent times are relative; after a week the date itself is clearer
export function relativeTime(ms: number, now = Date.now()) {
  const minutes = Math.floor((now - ms) / 60_000)
  if (minutes < 1) return 'Just now'
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days < 7) return `${days}d ago`
  const date = new Date(ms)
  return date.getFullYear() === new Date(now).getFullYear()
    ? dayFormat.format(date)
    : dateFormat.format(date)
}

// a span of days in the largest units that read naturally: "33 days",
// "5 months", "1 year 11 months"
export function formatDuration(days: number | null | undefined) {
  if (!Number.isFinite(days)) return MISSING
  const total = Math.max(0, Math.round(Number(days)))
  if (total < 60) return plural(total, 'day')
  const months = Math.round(total / 30.44)
  if (months < 12) return plural(months, 'month')
  const years = Math.floor(months / 12)
  const rest = months % 12
  return rest ? `${plural(years, 'year')} ${plural(rest, 'month')}` : plural(years, 'year')
}

export function plural(value: number | undefined, singular: string, pluralLabel = `${singular}s`) {
  return `${formatNumber(value || 0)} ${value === 1 ? singular : pluralLabel}`
}
