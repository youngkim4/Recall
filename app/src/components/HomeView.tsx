import type { Contact, Defaults, Job, ReportFile, ViewKey } from '../types'
import { contactTitle, formatNumber, reportTitle, shortDate, shortDateTime } from '../lib/format'

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
  defaults,
  contacts,
  reports,
  reportCount,
  namedCount,
  jobs,
  onViewChange,
  onSelectContact,
  onOpenReport,
}: HomeViewProps) {
  const topContacts = contacts.slice(0, 6)
  const recentReports = reports.filter((report) => report.kind === 'md').slice(0, 4)
  const activeJobs = jobs.filter((job) => job.status === 'queued' || job.status === 'running')
  const messages = contacts.reduce((sum, contact) => sum + Number(contact.message_count || 0), 0)

  return (
    <section className="home-view" aria-label="Home">
      <div className="home-hero">
        <div>
          <span className="eyebrow">Library</span>
          <h2>{formatNumber(messages)} messages indexed</h2>
          <p>
            {formatNumber(contacts.length)} conversations, {formatNumber(namedCount)} named contacts,{' '}
            {formatNumber(reportCount)} saved report groups.
          </p>
        </div>
        <div className="home-actions">
          <button type="button" className="button primary" onClick={() => onViewChange('ask')}>
            Open chat
          </button>
          <button type="button" className="button ghost" onClick={() => onViewChange('analyze')}>
            Analyze report
          </button>
        </div>
      </div>

      <div className="dashboard-grid">
        <section className="dashboard-panel">
          <div className="block-heading">
            <div>
              <h2>Ready State</h2>
              <p>Local data the app can use right now.</p>
            </div>
          </div>
          <div className="status-list">
            <StatusRow label="Messages CSV" value={defaults?.hasMessages ? 'Found' : 'Missing'} good={defaults?.hasMessages} />
            <StatusRow label="Messages database" value={defaults?.hasDb ? 'Found' : 'Path set'} good={defaults?.hasDb} />
            <StatusRow label="Contact names" value={formatNumber(namedCount)} good={namedCount > 0} />
            <StatusRow label="Active jobs" value={formatNumber(activeJobs.length)} good={activeJobs.length === 0} />
          </div>
        </section>

        <section className="dashboard-panel">
          <div className="block-heading">
            <div>
              <h2>Top Conversations</h2>
              <p>Fast entry points into analysis.</p>
            </div>
          </div>
          <div className="compact-list">
            {topContacts.map((contact) => (
              <button
                key={contact.chat_id}
                type="button"
                className="compact-row"
                onClick={() => onSelectContact(contact)}
              >
                <strong>{contactTitle(contact)}</strong>
                <span>{formatNumber(contact.message_count || 0)} messages</span>
              </button>
            ))}
          </div>
        </section>

        <section className="dashboard-panel">
          <div className="block-heading">
            <div>
              <h2>Recent Reports</h2>
              <p>Generated analysis you can reopen.</p>
            </div>
            <button type="button" className="button ghost" onClick={() => onViewChange('reports')}>
              All reports
            </button>
          </div>
          <div className="report-strip">
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
            {!recentReports.length ? (
              <div className="empty-state">
                <strong>No reports yet.</strong>
                <span>Run a report from Analyze after choosing a conversation.</span>
              </div>
            ) : null}
          </div>
        </section>

        <section className="dashboard-panel">
          <div className="block-heading">
            <div>
              <h2>Recent Jobs</h2>
              <p>Last report and export runs.</p>
            </div>
            <button type="button" className="button ghost" onClick={() => onViewChange('jobs')}>
              Jobs
            </button>
          </div>
          <div className="compact-list">
            {jobs.slice(0, 4).map((job) => (
              <div key={job.id} className="compact-row static">
                <strong>{job.action}</strong>
                <span>
                  {job.status} / {shortDate(job.updatedAt)}
                </span>
              </div>
            ))}
            {!jobs.length ? <div className="empty-inline">No jobs in this session.</div> : null}
          </div>
        </section>
      </div>
    </section>
  )
}

function StatusRow({ label, value, good }: { label: string; value: string; good?: boolean }) {
  return (
    <div className="status-row">
      <span>{label}</span>
      <strong className={good ? 'good' : ''}>{value}</strong>
    </div>
  )
}
