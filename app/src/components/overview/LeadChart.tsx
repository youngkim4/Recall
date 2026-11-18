import type { CSSProperties } from 'react'
import type { ActivityPoint } from '../../lib/activity'
import { peakAt, yearTotals } from '../../lib/leadChart'

type LeadChartProps = {
  months: ActivityPoint[]
}

// How the messages of the lead figure came: a bar for each year, the
// busiest one darker. Under the bars stand the first year, the busiest and
// the last, so the chart says where it starts, where it peaks and where it
// stops. (A line by month, a bar for each month and the running total were
// tried beside it and set aside: at this size they were noise, or said
// again what the figure says.)
export function LeadChart({ months }: LeadChartProps) {
  const years = yearTotals(months)
  if (years.length < 2) return null
  const largest = Math.max(1, ...years.map((year) => year.total))
  const peak = peakAt(years.map((year) => year.total))
  const named = (index: number) => index === 0 || index === years.length - 1 || index === peak
  return (
    <figure className="lead-chart" aria-hidden="true">
      <div className="lead-bars">
        {years.map((year, index) => (
          <span key={year.year} data-peak={index === peak || undefined} style={{ '--share': year.total / largest } as CSSProperties} />
        ))}
      </div>
      <figcaption className="lead-years">
        {years.map((year, index) => (
          <span key={year.year}>{named(index) ? year.year : ''}</span>
        ))}
      </figcaption>
    </figure>
  )
}
