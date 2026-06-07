import type { Contact, Defaults, Job, ReportFile, ViewKey } from '../types'
import { contactTitle, formatNumber, initials, reportTitle, shortDate, shortDateTime } from '../lib/format'

type HomeViewProps = {
  defaults: Defaults | null
  contacts: Contact[]
  reports: ReportFile[]
  reportCount: number
  namedCount: number
  jobs: Job[]
  onViewChange: (view: ViewKey) => void
  onSelectContact: (contact: Contact) => void
  onOpenReport: (report: ReportFile) => void
}

export function HomeView({
  contacts,
  reports,
  reportCount,
  namedCount,
  onViewChange,
  onSelectContact,
  onOpenReport,
}: HomeViewProps) {
  const people = contacts.slice(0, 9)
  const maxCount = Math.max(1, ...people.map((contact) => Number(contact.message_count || 0)))
  const recentReports = reports.filter((report) => report.kind === 'md').slice(0, 6)
  const messages = contacts.reduce((sum, contact) => sum + Number(contact.message_count || 0), 0)

  return (
    <section className="home-view" aria-label="Home">
      <div className="home-hero">
        <div>
          <span className="eyebrow">Library</span>
          <h2>{formatNumber(messages)} messages</h2>
          <p>
            Across {formatNumber(contacts.length)} conversations · {formatNumber(namedCount)} named ·{' '}
            {formatNumber(reportCount)} reports
          </p>
        </div>
        <div className="home-actions">
          <button type="button" className="button primary" onClick={() => onViewChange('ask')}>
            Open chat
          </button>
          <button type="button" className="button ghost" onClick={() => onViewChange('analyze')}>
            Analyze a conversation
          </button>
        </div>
      </div>

      <section className="home-section">
        <div className="home-section-head">
          <h3>People</h3>
          <button type="button" className="link-button" onClick={() => onViewChange('analyze')}>
            View all
          </button>
        </div>
        <div className="people-grid">
          {people.map((contact) => {
            const count = Number(contact.message_count || 0)
            const pct = Math.max(4, Math.round((count / maxCount) * 100))
            const range =
              contact.first_msg && contact.last_msg
                ? `${shortDate(contact.first_msg)} – ${shortDate(contact.last_msg)}`
                : contact.is_group
                  ? 'Group'
                  : 'Direct'
            return (
              <button
                key={contact.chat_id}
                type="button"
                className="person-card"
                onClick={() => onSelectContact(contact)}
              >
                <div className="person-top">
                  <span className="person-avatar" aria-hidden>
                    {initials(contactTitle(contact))}
                  </span>
                  <span className="person-id">
                    <strong>{contactTitle(contact)}</strong>
                    <span>{range}</span>
                  </span>
                </div>
                <div className="person-count">
                  {formatNumber(count)}
                  <small>messages</small>
                </div>
                <div className="person-bar">
                  <span style={{ width: `${pct}%` }} />
                </div>
              </button>
            )
          })}
          {!people.length ? (
            <div className="empty-state tall">
              <strong>No conversations yet.</strong>
              <span>Refresh the message export in Settings to load your archive.</span>
            </div>
          ) : null}
        </div>
      </section>

      <section className="home-section">
        <div className="home-section-head">
          <h3>Recent reports</h3>
          <button type="button" className="link-button" onClick={() => onViewChange('reports')}>
            All reports
          </button>
        </div>
        {recentReports.length ? (
          <div className="report-grid">
            {recentReports.map((report) => (
              <button
                key={report.path}
                type="button"
                className="report-card"
                onClick={() => onOpenReport(report)}
              >
                <strong>{reportTitle(report)}</strong>
                <span>{shortDateTime(report.updatedAt)}</span>
              </button>
            ))}
          </div>
        ) : (
          <div className="empty-state tall">
            <strong>No reports yet.</strong>
            <span>Pick a conversation in Analyze and run a report.</span>
          </div>
        )}
      </section>
    </section>
  )
}
