import { keyFigures } from '../../lib/reportNumbers'
import type { PreviewStats } from '../../types'

// What a report leads with: a few figures, a card each, its name over the
// number and one line under it.
export function ReportFigures({ stats }: { stats: PreviewStats }) {
  const figures = keyFigures(stats)
  if (!figures.length) return null
  return (
    <dl className="report-figures">
      {figures.map((figure) => (
        <div key={figure.label} className="report-figure card">
          <dt>{figure.label}</dt>
          <dd>{figure.value}</dd>
          {figure.note ? <dd className="report-figure-note">{figure.note}</dd> : null}
        </div>
      ))}
    </dl>
  )
}
