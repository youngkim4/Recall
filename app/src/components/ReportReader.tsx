import { useMemo, useState, type CSSProperties } from 'react'
import { ActivityChart } from './ActivityChart'
import { ResizeHandle } from './ResizeHandle'
import type { AnalysisEvent, AnalysisPayload, PreviewStats, ReportFile } from '../types'
import { displayContact, formatNumber, reportTitle, shortDate } from '../lib/format'

type ReportReaderProps = {
  reports: ReportFile[]
  selectedReport: ReportFile | null
  analysis: AnalysisPayload | null
  loading: boolean
  listWidth: number
  onListWidthChange: (value: number) => void
  onSelectReport: (report: ReportFile) => void
}

type ReportTab = 'overview' | 'events' | 'story' | 'files'

type ReportGroup = {
  key: string
  title: string
  contact: string
  updatedAt?: string
  readerReport: ReportFile
  mdReport?: ReportFile
  htmlReport?: ReportFile
  fileCount: number
}

type MetricEntry = {
  label: string
  value: number
  color: string
}

const CHART_COLORS = ['#c13c27', '#c2871d', '#5d7b67', '#4e6b88', '#8a586b', '#6f7d40', '#a8551f', '#6a5685']

const reportTabs: Array<{ key: ReportTab; label: string }> = [
  { key: 'overview', label: 'Overview' },
  { key: 'events', label: 'Timeline' },
  { key: 'story', label: 'Story' },
  { key: 'files', label: 'Files' },
]

export function ReportReader({
  reports,
  selectedReport,
  analysis,
  loading,
  listWidth,
  onListWidthChange,
  onSelectReport,
}: ReportReaderProps) {
  const [activeTab, setActiveTab] = useState<ReportTab>('overview')
  const groups = useMemo(() => groupReports(reports), [reports])
  const selectedGroupKey = selectedReport ? reportGroupKey(selectedReport) : ''
  const viewStyle = { '--report-list-width': `${listWidth}px` } as CSSProperties
  const tabCounts = useMemo(
    () => ({
      events: analysis?.events?.length || 0,
      files: analysis?.files?.length || 0,
    }),
    [analysis],
  )

  return (
    <section className="reports-view" style={viewStyle} aria-label="Reports">
      <aside className="report-list">
        <div className="pane-heading compact">
          <div>
            <h2>Reports</h2>
            <p>
              {formatNumber(groups.length)} reports / {formatNumber(reports.length)} files
            </p>
          </div>
        </div>
        <div className="report-items">
          {groups.map((group) => (
            <div key={group.key} className={`report-row report-group ${selectedGroupKey === group.key ? 'active' : ''}`}>
              <button
                type="button"
                className="report-group-main"
                onClick={() => {
                  void onSelectReport(group.readerReport)
                }}
              >
                <strong>{group.title}</strong>
                <span>
                  {shortDate(group.updatedAt)} / {displayContact(group.contact)} / {formatNumber(group.fileCount)} files
                </span>
              </button>
              <div className="report-format-actions" aria-label={`${group.title} files`}>
                {group.mdReport ? <ReportFileLink report={group.mdReport} label="MD" /> : null}
                {group.htmlReport ? <ReportFileLink report={group.htmlReport} label="HTML" /> : null}
              </div>
            </div>
          ))}
        </div>
      </aside>

      <ResizeHandle
        label="Resize reports list"
        value={listWidth}
        min={280}
        max={560}
        onChange={onListWidthChange}
      />

      <article className="report-reader">
        <div className="detail-header report-reader-header">
          <div>
            <span className="eyebrow">Report</span>
            <h1>{analysis?.contactDisplayName || reportTitle(selectedReport || undefined)}</h1>
          </div>
          <span className="date-pill">
            {analysis?.stats?.firstTimestamp ? shortDate(analysis.stats.firstTimestamp) : '--'} to{' '}
            {analysis?.stats?.lastTimestamp ? shortDate(analysis.stats.lastTimestamp) : '--'}
          </span>
        </div>

        {loading ? (
          <div className="empty-state tall">
            <strong>Loading report...</strong>
            <span>Reading the saved analysis payload.</span>
          </div>
        ) : analysis ? (
          <>
            <nav className="report-tabs" aria-label="Report sections">
              {reportTabs.map((tab) => {
                const count = tab.key === 'story' || tab.key === 'overview' ? null : tabCounts[tab.key]
                return (
                  <button
                    key={tab.key}
                    type="button"
                    className={activeTab === tab.key ? 'active' : ''}
                    onClick={() => setActiveTab(tab.key)}
                  >
                    {tab.label}
                    {count ? <span>{formatNumber(count)}</span> : null}
                  </button>
                )
              })}
            </nav>
            <div className="report-content">{renderReportSection(activeTab, analysis)}</div>
          </>
        ) : (
          <div className="empty-state tall">
            <strong>No report selected.</strong>
            <span>Choose a saved report from the left.</span>
          </div>
        )}
      </article>
    </section>
  )
}

function ReportFileLink({ report, label }: { report: ReportFile; label: string }) {
  return (
    <a href={reportHref(report)} target="_blank" rel="noreferrer">
      {label}
    </a>
  )
}

function reportHref(report: ReportFile) {
  return `/api/report?path=${encodeURIComponent(report.path)}`
}

function reportGroupKey(report: ReportFile) {
  return report.contact || report.displayName || report.path.replace(/\.(html|md)$/i, '')
}

function groupReports(reports: ReportFile[]) {
  const map = new Map<string, ReportFile[]>()
  for (const report of reports) {
    const key = reportGroupKey(report)
    map.set(key, [...(map.get(key) || []), report])
  }

  return Array.from(map.entries())
    .map(([key, items]): ReportGroup => {
      const sorted = [...items].sort((left, right) =>
        String(right.updatedAt || '').localeCompare(String(left.updatedAt || '')),
      )
      const mdReport = sorted.find((report) => report.kind === 'md')
      const htmlReport = sorted.find((report) => report.kind === 'html')
      const readerReport = mdReport || sorted[0]
      const latest = sorted[0]
      return {
        key,
        title: reportTitle(readerReport),
        contact: readerReport.contact,
        updatedAt: latest.updatedAt || readerReport.updatedAt,
        readerReport,
        mdReport,
        htmlReport,
        fileCount: items.length,
      }
    })
    .sort((left, right) => String(right.updatedAt || '').localeCompare(String(left.updatedAt || '')))
}

function renderReportSection(activeTab: ReportTab, analysis: AnalysisPayload) {
  if (activeTab === 'overview') {
    return <ReportOverview analysis={analysis} />
  }

  if (activeTab === 'story') {
    return (
      <section className="report-prose">
        {analysis.summaryHtml ? (
          <div dangerouslySetInnerHTML={{ __html: analysis.summaryHtml }} />
        ) : (
          <p>No summary section was found in this report.</p>
        )}
      </section>
    )
  }

  if (activeTab === 'events') {
    return (
      <section className="visual-section">
        <SectionHeading title="Timeline" meta={`${formatNumber(analysis.events?.length || 0)} extracted moments`} />
        <EventCategoryBars events={analysis.events || []} />
        <div className="event-list full compact-events">
          {(analysis.events || []).map((event) => (
            <div key={`${event.date}-${event.title}-${event.detail}`} className="event-row">
              <span>{shortDate(event.date)}</span>
              <div>
                <strong>{event.title || 'Untitled moment'}</strong>
                <p>{event.detail}</p>
                {event.quote ? <blockquote>{event.quote}</blockquote> : null}
              </div>
            </div>
          ))}
          {!analysis.events?.length ? <EmptyInline message="No event timeline was generated." /> : null}
        </div>
      </section>
    )
  }

  return (
    <section>
      <SectionHeading title="Files" meta={`${formatNumber(analysis.files?.length || 0)} outputs`} />
      <div className="file-list">
        {(analysis.files || []).map((file) => (
          <a
            key={file.path}
            className="file-row"
            href={`/api/report?path=${encodeURIComponent(file.path)}`}
            target="_blank"
            rel="noreferrer"
          >
            <div>
              <strong>{file.label || file.name}</strong>
              <span>{file.name}</span>
            </div>
            <span>{file.kind.toUpperCase()}</span>
          </a>
        ))}
        {!analysis.files?.length ? <EmptyInline message="No local report files were found." /> : null}
      </div>
    </section>
  )
}

function ReportOverview({ analysis }: { analysis: AnalysisPayload }) {
  const summary = compactSummary(analysis.summary || analysis.summaryHtml || '')
  const eventValues = eventCategoryValues(analysis.events || [])
  const attachmentValues = attachmentTypeValues(analysis)
  const attachmentTotal = metricTotal(attachmentValues, mediaStats(analysis, 'attachments').total)
  return (
    <section className="report-overview">
      <StatsMosaic stats={analysis.stats || {}} />
      <div className="visual-card activity-card activity-overview-card">
        <SectionHeading title="Activity" meta={`${formatNumber(analysis.monthly.length)} months`} />
        <div className="activity-overview-grid">
          <ActivityChart data={analysis.monthly} />
          <MonthlyBreakdown monthly={analysis.monthly} />
        </div>
      </div>
      <div className="visual-card split-card">
        <SectionHeading
          title="Balance"
          meta={`${formatNumber(
            Number(analysis.stats?.sentCount || 0) + Number(analysis.stats?.receivedCount || 0),
          )} messages`}
        />
        <DirectionSplit stats={analysis.stats || {}} />
      </div>
      <div className="visual-grid two mix-grid">
        <div className="visual-card event-card">
          <SectionHeading title="Event Mix" meta={`${formatNumber(analysis.events?.length || 0)} moments`} />
          <VisualBreakdown values={eventValues} empty="No event categories yet." showLegend={false} />
        </div>
        <div className="visual-card attachments-card">
          <SectionHeading title="Attachments" meta={`100% = ${formatNumber(attachmentTotal)} files`} />
          <MediaBreakdown analysis={analysis} />
        </div>
      </div>
      <div className="visual-grid two supporting-grid">
        <div className="visual-card signals-card">
          <SectionHeading title="Signals" meta="Compact pattern readout" />
          <PatternVisualGrid patterns={(analysis.patterns || []).slice(0, 4)} compact />
        </div>
        <div className="visual-card media-detail-card">
          <SectionHeading
            title="Tapbacks"
            meta={`${formatNumber(metricTotal(reactionTypeValues(analysis), mediaStats(analysis, 'reactions').total))} reactions`}
          />
          <OverviewMediaSignals analysis={analysis} />
        </div>
      </div>
      {summary ? (
        <div className="visual-card summary-card">
          <SectionHeading title="Short Read" meta="Condensed narrative" />
          <p>{summary}</p>
        </div>
      ) : null}
    </section>
  )
}

function MonthlyBreakdown({
  monthly,
}: {
  monthly: AnalysisPayload['monthly']
}) {
  if (!monthly.length) return <EmptyInline message="No monthly activity found." />
  return (
    <div className="monthly-table overview-monthly-table" aria-label="Monthly activity breakdown">
      {monthly.map((point) => (
        <div key={point.month}>
          <span>{point.month}</span>
          <strong>{formatNumber(point.total)}</strong>
          <span>{formatNumber(point.sent)} sent</span>
          <span>{formatNumber(point.received)} received</span>
        </div>
      ))}
    </div>
  )
}

function OverviewMediaSignals({ analysis }: { analysis: AnalysisPayload }) {
  const reactionStats = mediaStats(analysis, 'reactions')
  const reactionValues = reactionTypeValues(analysis)
  return (
    <div className="overview-media-signals">
      <MetricBars values={reactionValues} empty="No tapbacks found." />
      <MiniMeter
        label="Direction"
        leftLabel="Sent"
        leftValue={Number(reactionStats.sent_reactions || 0)}
        rightLabel="Received"
        rightValue={Number(reactionStats.received_reactions || 0)}
      />
    </div>
  )
}

function StatsMosaic({ stats }: { stats: PreviewStats }) {
  const total = Number(stats.totalMessages || 0)
  const sent = Number(stats.sentCount || 0)
  const received = Number(stats.receivedCount || 0)
  const activeDays = Number(stats.activeDays || 0)
  return (
    <div className="stats-mosaic">
      <VisualStat label="Messages" value={formatNumber(total)} />
      <VisualStat label="Sent" value={formatNumber(sent)} tone="teal" />
      <VisualStat label="Received" value={formatNumber(received)} tone="violet" />
      <VisualStat label="Active days" value={formatNumber(activeDays)} />
      <VisualStat label="Avg / day" value={formatDecimal(stats.avgMessagesPerDay)} />
      <VisualStat label="Busiest day" value={stats.busiestDay ? shortDate(stats.busiestDay) : '--'} />
      <VisualStat label="Busiest volume" value={formatNumber(stats.busiestDayCount || 0)} tone="amber" />
      <VisualStat label="Longest gap" value={`${formatDecimal(stats.longestGapDays)}d`} />
    </div>
  )
}

function VisualStat({ label, value, tone = '' }: { label: string; value: string; tone?: string }) {
  return (
    <div className={`visual-stat ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function DirectionSplit({ stats }: { stats: PreviewStats }) {
  const sent = Number(stats.sentCount || 0)
  const received = Number(stats.receivedCount || 0)
  const total = Math.max(1, sent + received)
  const sentPct = Math.round((sent / total) * 100)
  const receivedPct = 100 - sentPct
  const difference = Math.abs(sent - received)
  const leader = sent === received ? 'Even split' : sent > received ? 'Sent led' : 'Received led'
  const leaderTone = sent >= received ? 'teal' : 'violet'
  const ratio = received ? `${formatDecimal(sent / received)}:1` : sent ? 'All sent' : '--'

  return (
    <div className="balance-split">
      <div className="balance-copy">
        <span className={`balance-kicker ${leaderTone}`}>{leader}</span>
        <strong>{difference ? formatNumber(difference) : 'No gap'}</strong>
        <p>
          {formatNumber(sent)} sent / {formatNumber(received)} received.
        </p>
      </div>
      <div className="balance-panel">
        <div className="balance-meter" aria-label={`Sent ${sentPct}%, received ${receivedPct}%`}>
          <span
            className="balance-segment sent"
            style={{ width: `${sentPct}%` }}
            data-tooltip={`${formatNumber(sent)} sent / ${sentPct}%`}
          />
          <span
            className="balance-segment received"
            style={{ width: `${receivedPct}%` }}
            data-tooltip={`${formatNumber(received)} received / ${receivedPct}%`}
          />
        </div>
        <div className="balance-grid">
          <BalanceMetric label="Sent" value={formatNumber(sent)} meta={`${sentPct}%`} tone="teal" />
          <BalanceMetric label="Received" value={formatNumber(received)} meta={`${receivedPct}%`} tone="violet" />
          <BalanceMetric label="Ratio" value={ratio} meta="sent : received" />
        </div>
      </div>
    </div>
  )
}

function BalanceMetric({
  label,
  value,
  meta,
  tone = '',
}: {
  label: string
  value: string
  meta: string
  tone?: string
}) {
  return (
    <div className={`balance-metric ${tone}`} data-tooltip={`${label}: ${value} (${meta})`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{meta}</small>
    </div>
  )
}

function EventCategoryBars({ events }: { events: AnalysisEvent[] }) {
  return <MetricBars values={eventCategoryValues(events)} empty="No event categories yet." />
}

function VisualBreakdown({
  values,
  empty = 'None found.',
  centerLabel,
  centerSubLabel,
  showLegend = true,
}: {
  values: Record<string, number>
  empty?: string
  centerLabel?: string
  centerSubLabel?: string
  showLegend?: boolean
}) {
  const entries = metricEntries(values, 6)
  const total = metricTotal(values)
  if (!entries.length) return <EmptyInline message={empty} />
  return (
    <div className="visual-breakdown">
      <DonutChart
        entries={entries}
        centerLabel={centerLabel}
        centerSubLabel={centerSubLabel}
        showLegend={showLegend}
      />
      <MetricBars values={Object.fromEntries(entries.map((entry) => [entry.label, entry.value]))} denominator={total} />
    </div>
  )
}

function MediaBreakdown({ analysis }: { analysis: AnalysisPayload }) {
  const attachmentStats = mediaStats(analysis, 'attachments')
  const reactionStats = mediaStats(analysis, 'reactions')
  const attachmentValues = attachmentTypeValues(analysis)
  const reactionValues = reactionTypeValues(analysis)
  const attachmentTotal = metricTotal(attachmentValues, attachmentStats.total)
  const reactionTotal = metricTotal(reactionValues, reactionStats.total)
  const sentAttachments = Number(attachmentStats.sent_attachments || 0)
  const receivedAttachments = Number(attachmentStats.received_attachments || 0)

  if (!attachmentTotal && !reactionTotal) return <EmptyInline message="No media was found." />

  return (
    <div className="media-breakdown">
      <MetricBars values={attachmentValues} empty="No attachments found." denominator={attachmentTotal} />
      <div className="media-context">
        <MiniMeter
          label="Direction"
          leftLabel="Sent"
          leftValue={sentAttachments}
          rightLabel="Received"
          rightValue={receivedAttachments}
        />
        <div className="media-mini-stats">
          <MiniStat label="Tapbacks" value={formatNumber(reactionTotal)} />
          <MiniStat label="Top type" value={topMetricLabel(attachmentValues)} />
        </div>
      </div>
    </div>
  )
}

function MiniMeter({
  label,
  leftLabel,
  leftValue,
  rightLabel,
  rightValue,
}: {
  label: string
  leftLabel: string
  leftValue: number
  rightLabel: string
  rightValue: number
}) {
  const total = Math.max(1, leftValue + rightValue)
  const leftPct = Math.round((leftValue / total) * 100)
  const rightPct = 100 - leftPct
  return (
    <div className="mini-meter">
      <div>
        <span>{label}</span>
        <strong>
          {formatNumber(leftValue)} / {formatNumber(rightValue)}
        </strong>
      </div>
      <div className="balance-meter compact" aria-label={`${leftLabel} ${leftPct}%, ${rightLabel} ${rightPct}%`}>
        <span
          className="balance-segment sent"
          style={{ width: `${leftPct}%` }}
          data-tooltip={`${leftLabel}: ${formatNumber(leftValue)} / ${leftPct}%`}
        />
        <span
          className="balance-segment received"
          style={{ width: `${rightPct}%` }}
          data-tooltip={`${rightLabel}: ${formatNumber(rightValue)} / ${rightPct}%`}
        />
      </div>
      <small>
        {leftLabel} {leftPct}% / {rightLabel} {rightPct}%
      </small>
    </div>
  )
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function DonutChart({
  entries,
  centerLabel,
  centerSubLabel = 'total',
  showLegend = true,
}: {
  entries: MetricEntry[]
  centerLabel?: string
  centerSubLabel?: string
  showLegend?: boolean
}) {
  const [activeLabel, setActiveLabel] = useState<string | null>(null)
  const total = entries.reduce((sum, entry) => sum + entry.value, 0)
  const safeTotal = Math.max(1, total)
  const activeEntry = activeLabel ? entries.find((entry) => entry.label === activeLabel) : null
  const activePercent = activeEntry ? Math.round((activeEntry.value / safeTotal) * 100) : null
  const segments = entries.map((entry, index) => {
    const start = entries.slice(0, index).reduce((sum, item) => sum + (item.value / safeTotal) * 100, 0)
    return {
      entry,
      share: (entry.value / safeTotal) * 100,
      dashOffset: -start,
    }
  })

  return (
    <div className="donut-wrap" onPointerLeave={() => setActiveLabel(null)}>
      <div className="donut-chart" aria-label={entries.map((entry) => `${entry.label}: ${formatNumber(entry.value)}`).join(', ')}>
        <svg viewBox="0 0 100 100" role="img" aria-label="Interactive category breakdown">
          {segments.map(({ entry, share, dashOffset }) => {
            return (
              <circle
                key={entry.label}
                className={`donut-segment ${activeLabel === entry.label ? 'active' : ''}`}
                cx="50"
                cy="50"
                r="38"
                fill="none"
                pathLength="100"
                stroke={entry.color}
                strokeDasharray={`${share} ${100 - share}`}
                strokeDashoffset={dashOffset}
                onPointerEnter={() => setActiveLabel(entry.label)}
                onFocus={() => setActiveLabel(entry.label)}
                onBlur={() => setActiveLabel(null)}
                tabIndex={0}
              />
            )
          })}
        </svg>
        <div className="donut-hole">
          <strong>{activeEntry ? `${activePercent}%` : centerLabel || formatNumber(total)}</strong>
          <span>{activeEntry ? activeEntry.label : centerSubLabel}</span>
        </div>
        {activeEntry ? (
          <div className="chart-popover donut-popover">
            <strong>{activeEntry.label}</strong>
            <span>
              {formatNumber(activeEntry.value)} / {activePercent}%
            </span>
          </div>
        ) : null}
      </div>
      {showLegend ? (
        <div className="donut-legend">
          {entries.slice(0, 4).map((entry) => (
            <div
              key={entry.label}
              onPointerEnter={() => setActiveLabel(entry.label)}
              onPointerLeave={() => setActiveLabel(null)}
            >
              <span style={{ '--dot-color': entry.color } as CSSProperties} />
              <small>{entry.label}</small>
              <strong>{Math.round((entry.value / safeTotal) * 100)}%</strong>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  )
}

function MetricBars({
  values,
  empty = 'None found.',
  denominator,
}: {
  values: Record<string, number>
  empty?: string
  denominator?: number
}) {
  const entries = metricEntries(values, 8)
  const total = Math.max(1, Number(denominator || metricTotal(values)))

  if (!entries.length) return <EmptyInline message={empty} />

  return (
    <div className="metric-bars">
      {entries.map((entry, index) => (
        <div
          key={entry.label}
          className="metric-bar-row"
          data-tooltip={`${entry.label}: ${formatNumber(entry.value)} (${Math.round((entry.value / total) * 100)}% of total)`}
        >
          <span>{entry.label}</span>
          <div>
            <i
              style={{
                width: `${Math.max(5, (entry.value / total) * 100)}%`,
                background: entry.color,
                animationDelay: `${index * 40}ms`,
              }}
            />
          </div>
          <strong>{formatNumber(entry.value)}</strong>
        </div>
      ))}
    </div>
  )
}

function PatternVisualGrid({
  patterns,
  compact = false,
}: {
  patterns: Array<{ label: string; value: string; detail: string }>
  compact?: boolean
}) {
  if (!patterns.length) return <EmptyInline message="No pattern summary was generated." />
  return (
    <div className={`pattern-visual-grid ${compact ? 'compact' : ''}`}>
      {patterns.map((pattern, index) => (
        <div key={pattern.label} className="pattern-visual">
          <span>{pattern.label}</span>
          <strong>{displayPatternValue(pattern.value)}</strong>
          {!compact ? <p>{pattern.detail}</p> : null}
          <i style={{ width: `${Math.max(22, 100 - index * 12)}%` }} />
        </div>
      ))}
    </div>
  )
}

function SectionHeading({ title, meta }: { title: string; meta: string }) {
  return (
    <div className="block-heading">
      <div>
        <h2>{title}</h2>
        <p>{meta}</p>
      </div>
    </div>
  )
}

function EmptyInline({ message }: { message: string }) {
  return <div className="empty-inline">{message}</div>
}

function humanizeKey(key: string) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function eventCategoryValues(events: AnalysisEvent[]) {
  return events.reduce<Record<string, number>>((counts, event) => {
    const key = humanizeKey(event.category || 'uncategorized')
    counts[key] = (counts[key] || 0) + 1
    return counts
  }, {})
}

function metricEntries(values: Record<string, number>, limit: number): MetricEntry[] {
  return Object.entries(values)
    .map(([key, value]) => [humanizeKey(key), Number(value || 0)] as const)
    .filter(([, value]) => value > 0)
    .sort((left, right) => right[1] - left[1])
    .slice(0, limit)
    .map(([label, value], index) => ({
      label,
      value,
      color: CHART_COLORS[index % CHART_COLORS.length],
    }))
}

function mediaStats(analysis: AnalysisPayload, key: 'attachments' | 'reactions'): Record<string, number> {
  return analysis.media?.[key] || analysis.stats?.[key] || {}
}

function attachmentTypeValues(analysis: AnalysisPayload) {
  return metricSubset(mediaStats(analysis, 'attachments'), {
    photos: 'Photos',
    videos: 'Videos',
    audio: 'Audio',
    gifs: 'GIFs',
    documents: 'Documents',
    other: 'Other',
  })
}

function reactionTypeValues(analysis: AnalysisPayload) {
  return metricSubset(mediaStats(analysis, 'reactions'), {
    loves: 'Loved',
    likes: 'Liked',
    dislikes: 'Disliked',
    laughs: 'Laughed',
    emphasis: 'Emphasized',
    questions: 'Questioned',
  })
}

function metricSubset(values: Record<string, number>, labels: Record<string, string>) {
  return Object.fromEntries(
    Object.entries(labels)
      .map(([key, label]) => [label, Number(values[key] || 0)] as const)
      .filter(([, value]) => value > 0),
  )
}

function metricTotal(values: Record<string, number>, fallback?: unknown) {
  const total = Number(fallback || 0)
  return total > 0 ? total : Object.values(values).reduce((sum, value) => sum + Number(value || 0), 0)
}

function topMetricLabel(values: Record<string, number>) {
  const top = Object.entries(values).sort((left, right) => Number(right[1]) - Number(left[1]))[0]
  return top ? top[0] : '--'
}

function formatDecimal(value?: number) {
  const number = Number(value || 0)
  return Number.isFinite(number) ? number.toFixed(number >= 10 ? 1 : 2).replace(/\.0$/, '') : '--'
}

function compactSummary(value: string) {
  const text = value
    .replace(/<[^>]+>/g, ' ')
    .replace(/[#*_`>-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
  if (!text) return ''
  return text.length > 420 ? `${text.slice(0, 420).trim()}...` : text
}

function displayPatternValue(value: string) {
  const text = String(value || '').trim()
  if (/^\d{4}-\d{2}-\d{2}T/.test(text)) return shortDate(text)
  return text
}
