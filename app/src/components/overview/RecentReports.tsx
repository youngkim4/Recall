import { reportTitle } from '../../lib/conversations'
import { parseDate, relativeTime } from '../../lib/format'
import type { ReportFile } from '../../types'
import { Truncate } from '../ui/Truncate'

type RecentReportsProps = {
  reports: ReportFile[]
  onOpenReport: (report: ReportFile) => void
  onViewAll: () => void
  onChooseConversation: () => void
}

const SHOWN = 6

function writtenOn(report: ReportFile) {
  const written = parseDate(report.updatedAt)
  return written ? relativeTime(written.valueOf()) : ''
}

export function RecentReports({ reports, onOpenReport, onViewAll, onChooseConversation }: RecentReportsProps) {
  // a report is written as Markdown; its other files are companions to that one
  const recent = reports.filter((report) => report.kind === 'md').slice(0, SHOWN)

  return (
    <section className="section ruled" aria-labelledby="reports-title">
      <div className="section-head">
        <h2 id="reports-title">Recent reports</h2>
        {recent.length ? (
          <span className="section-links">
            <button type="button" className="link" onClick={onViewAll}>
              All reports
            </button>
          </span>
        ) : null}
      </div>
      {recent.length ? (
        <ul className="rows ruled" role="list">
          {recent.map((report) => (
            <li key={report.path}>
              <button type="button" className="row" onClick={() => onOpenReport(report)}>
                <span className="row-main">
                  <Truncate className="row-title">{reportTitle(report)}</Truncate>
                </span>
                <span className="row-end">{writtenOn(report)}</span>
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <div className="empty">
          <p>No reports yet</p>
          <button type="button" className="btn" onClick={onChooseConversation}>
            Analyze a conversation
          </button>
        </div>
      )}
    </section>
  )
}
