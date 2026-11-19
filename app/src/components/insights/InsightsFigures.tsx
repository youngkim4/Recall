import { formatDate, formatNumber, formatPercent, plural } from '../../lib/format'
import type { StatsPayload } from '../../lib/insights'

type InsightsFiguresProps = {
  stats: StatsPayload | null
  // the count by person or by conversation, for how many there are
  people: StatsPayload | null
  peopleBy: 'person' | 'conversation'
}

// The count in four figures, after Plausible's row of figures and the
// Overview's tiles: each a name, the figure and one line, set apart by a
// hairline. They follow the filters and the words.
export function InsightsFigures({ stats, people, peopleBy }: InsightsFiguresProps) {
  const totals = stats?.totals
  const among = people?.rowsTotal
    ? peopleBy === 'person'
      ? `with ${plural(people.rowsTotal, 'person', 'people')}`
      : `in ${plural(people.rowsTotal, 'conversation')}`
    : ''
  const share = (part: number) => (totals?.total ? `${formatPercent(part / totals.total)} of them` : '')
  const figures = [
    {
      label: 'Messages',
      value: totals ? formatNumber(totals.total) : '',
      note: totals?.total ? among : 'None match',
    },
    {
      label: 'You sent',
      value: totals ? formatNumber(totals.sent) : '',
      note: totals ? share(totals.sent) : '',
    },
    {
      label: 'They sent',
      value: totals ? formatNumber(totals.received) : '',
      note: totals ? share(totals.received) : '',
    },
    {
      label: 'From',
      value: stats?.first ? formatDate(stats.first) : totals ? 'No days' : '',
      note: stats?.last ? `to ${formatDate(stats.last)}` : '',
    },
  ]
  return (
    <div className="tiles insights-figures" aria-busy={!stats || undefined}>
      {figures.map((figure) => (
        <div key={figure.label} className="tile">
          <span className="tile-label">{figure.label}</span>
          <span className="tile-value swap" key={figure.value}>
            {figure.value || <span className="skeleton">000,000</span>}
          </span>
          <span className="tile-note">{figure.note || ' '}</span>
        </div>
      ))}
    </div>
  )
}
