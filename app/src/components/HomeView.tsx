import type { Contact, Defaults, Job, ReportFile, ViewKey } from '../types'
import { contactTitle, formatNumber, reportTitle, shortDateTime } from '../lib/format'
import { BarList } from './BarList'
import { Donut } from './Donut'

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

const DONUT_COLORS = [
  'var(--data-indigo)',
  'var(--data-blue)',
  'var(--data-teal)',
  'var(--data-amber)',
  'var(--data-rose)',
  'var(--data-violet)',
]
const OTHER_COLOR = '#c7bca8'

export function HomeView({
  contacts,
  reports,
  reportCount,
  namedCount,
  onViewChange,
  onSelectContact,
  onOpenReport,
}: HomeViewProps) {
  const ranked = [...contacts].sort(
    (a, b) => Number(b.message_count || 0) - Number(a.message_count || 0),
  )
  const totalMessages = contacts.reduce((sum, contact) => sum + Number(contact.message_count || 0), 0)

  const barItems = ranked.slice(0, 8).map((contact) => ({
    key: contact.chat_id,
    label: contactTitle(contact),
    value: Number(contact.message_count || 0),
    onClick: () => onSelectContact(contact),
  }))

  const topSix = ranked.slice(0, 6)
  const otherTotal = ranked.slice(6).reduce((sum, contact) => sum + Number(contact.message_count || 0), 0)
  const donutSegments = [
    ...topSix.map((contact, index) => ({
      key: contact.chat_id,
      label: contactTitle(contact),
      value: Number(contact.message_count || 0),
      color: DONUT_COLORS[index % DONUT_COLORS.length],
    })),
    ...(otherTotal > 0
      ? [{ key: '__other', label: `Other (${ranked.length - 6})`, value: otherTotal, color: OTHER_COLOR }]
      : []),
  ]
  const sharePercent = (value: number) =>
    totalMessages > 0 ? `${Math.round((value / totalMessages) * 100)}%` : '0%'

  const recentReports = reports.filter((report) => report.kind === 'md').slice(0, 6)

  return (
    <section className="home-view" aria-label="Overview">
      <div className="home-hero">
        <div>
          <span className="eyebrow">Library</span>
          <h2>Overview</h2>
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

      <div className="kpi-strip">
        <div className="kpi-card">
          <span className="kpi-label">Messages</span>
          <span className="kpi-value">{formatNumber(totalMessages)}</span>
          <span className="kpi-foot">
            <span className="kpi-sub">across {formatNumber(contacts.length)} conversations</span>
          </span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Conversations</span>
          <span className="kpi-value">{formatNumber(contacts.length)}</span>
          <span className="kpi-foot">
            <span className="kpi-sub">{formatNumber(namedCount)} named</span>
          </span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Top conversation</span>
          <span className="kpi-value">{ranked[0] ? formatNumber(Number(ranked[0].message_count || 0)) : '0'}</span>
          <span className="kpi-foot">
            <span className="kpi-sub">{ranked[0] ? contactTitle(ranked[0]) : 'No data'}</span>
          </span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Reports</span>
          <span className="kpi-value">{formatNumber(reportCount)}</span>
          <span className="kpi-foot">
            <span className="kpi-sub">{formatNumber(reports.length)} files</span>
          </span>
        </div>
      </div>

      <div className="ov-grid">
        <section className="panel">
          <div className="panel-head">
            <h3>Most messages</h3>
            <button type="button" className="link-button" onClick={() => onViewChange('analyze')}>
              View all
            </button>
          </div>
          {barItems.length ? (
            <BarList items={barItems} formatValue={formatNumber} />
          ) : (
            <div className="empty-state tall">
              <strong>No conversations yet.</strong>
              <span>Refresh the message export in Settings to load your archive.</span>
            </div>
          )}
        </section>

        <section className="panel">
          <div className="panel-head">
            <h3>Share of messages</h3>
          </div>
          {donutSegments.length ? (
            <Donut
              segments={donutSegments}
              centerLabel={formatNumber(totalMessages)}
              centerSub="messages"
              formatValue={sharePercent}
            />
          ) : (
            <div className="empty-state tall">
              <strong>No data yet.</strong>
            </div>
          )}
        </section>
      </div>

      <section className="panel" style={{ marginTop: 16 }}>
        <div className="panel-head">
          <h3>Recent reports</h3>
          <button type="button" className="link-button" onClick={() => onViewChange('reports')}>
            All reports
          </button>
        </div>
        {recentReports.length ? (
          <div className="ov-reports">
            {recentReports.map((report) => (
              <button
                key={report.path}
                type="button"
                className="ov-report-row"
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
