import type { ActivityPoint } from '../../lib/activity'
import { sparkArea, sparkLine } from '../../lib/spark'

type LeadChartProps = {
  months: ActivityPoint[]
}

// the box the line is drawn in; it is stretched to its place
const BOX = { width: 400, height: 56 }

// How the messages of the lead figure came: a line through the months, the
// first year under its start and the last under its end. (A bar for each
// year, a bar for each month and the running total were tried beside the
// figure; the user kept the line.)
export function LeadChart({ months }: LeadChartProps) {
  if (months.length < 2) return null
  const totals = months.map((point) => point.total)
  return (
    <figure className="lead-chart" aria-hidden="true">
      <svg viewBox={`0 0 ${BOX.width} ${BOX.height}`} preserveAspectRatio="none">
        <path className="lead-area" d={sparkArea(totals, BOX.width, BOX.height)} />
        <path className="lead-line" d={sparkLine(totals, BOX.width, BOX.height)} />
      </svg>
      <figcaption>
        <span>{months[0].key.slice(0, 4)}</span>
        <span>{months.at(-1)!.key.slice(0, 4)}</span>
      </figcaption>
    </figure>
  )
}
