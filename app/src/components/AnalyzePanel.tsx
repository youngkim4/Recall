import { useState } from 'react'
import { ActivityChart } from './ActivityChart'
import { SparkIcon } from './Icons'
import { DateField } from './DateField'
import type { Contact, Defaults, Dynamics, Job, PreviewPayload, ReportFile } from '../types'
import {
  contactTitle,
  formatMoney,
  formatNumber,
  isOutbound,
  plural,
  shortDate,
} from '../lib/format'
import { modelLabel } from '../lib/models'

type AnalyzePanelProps = {
  contact: Contact | null
  defaults: Defaults | null
  model: string
  scope: { since: string; until: string }
  preview: PreviewPayload | null
  reports: ReportFile[]
  loadingPreview: boolean
  runningJob: Job | null
  onRunReport: () => Promise<void>
  onOpenReport: (report: ReportFile) => void
  onViewReports: () => void
  onModelChange: (model: string) => void
  onScopeChange: (scope: { since: string; until: string }) => void
}

export function AnalyzePanel({
  contact,
  defaults,
  model,
  scope,
  preview,
  reports,
  loadingPreview,
  runningJob,
  onRunReport,
  onOpenReport,
  onViewReports,
  onModelChange,
  onScopeChange,
}: AnalyzePanelProps) {
  const contactReports = contact
    ? reports.filter((report) => report.contact === contact.chat_id || report.displayName === contactTitle(contact))
    : []
  const latestReport =
    [...contactReports].sort((left, right) => String(right.updatedAt || '').localeCompare(String(left.updatedAt || ''))).find(
      (report) => report.kind === 'md',
    ) || contactReports[0]
  const stats = preview?.stats
  const canRun = Boolean(contact && defaults?.messagesPath && defaults?.outDir)
  const jobBusy = runningJob?.status === 'queued' || runningJob?.status === 'running'

  return (
    <section className="detail-pane" aria-label="Analyze conversation">
      <div className="detail-header">
        <div>
          <span className="eyebrow">Thread</span>
          <h1>{contactTitle(contact || undefined)}</h1>
        </div>
        <span className="date-pill">
          {stats?.firstTimestamp || contact?.first_msg ? shortDate(stats?.firstTimestamp || contact?.first_msg) : '--'} to{' '}
          {stats?.lastTimestamp || contact?.last_msg ? shortDate(stats?.lastTimestamp || contact?.last_msg) : '--'}
        </span>
      </div>

      <div className="action-row">
        <div>
          <strong>Latest report</strong>
          <span>
            {latestReport
              ? `${latestReport.displayName || contactTitle(contact || undefined)} / ${shortDate(latestReport.updatedAt)}`
              : 'No saved report for this thread.'}
          </span>
        </div>
        <div className="row-actions">
          {latestReport ? (
            <button
              type="button"
              className="button secondary"
              onClick={() => {
                onOpenReport(latestReport)
              }}
            >
              Open report
            </button>
          ) : null}
          <button type="button" className="button ghost" onClick={onViewReports}>
            All reports
          </button>
        </div>
      </div>

      <div className="settings-strip">
        <label>
          <span>Model</span>
          <select value={model} onChange={(event) => onModelChange(event.target.value)}>
            {(defaults?.models || [model]).map((option) => (
              <option key={option} value={option}>
                {modelLabel(option)}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Start date</span>
          <DateField
            value={scope.since}
            ariaLabel="Start date"
            onChange={(since) => onScopeChange({ ...scope, since })}
          />
        </label>
        <label>
          <span>End date</span>
          <DateField
            value={scope.until}
            ariaLabel="End date"
            onChange={(until) => onScopeChange({ ...scope, until })}
          />
        </label>
      </div>

      <div className="preview-block">
        <div className="block-heading">
          <div>
            <h2>Preview</h2>
            <p>
              {loadingPreview
                ? 'Calculating...'
                : stats
                  ? `${plural(stats.totalMessages, 'message')} across ${formatNumber(stats.activeDays)} active days.`
                  : 'Choose a conversation.'}
            </p>
          </div>
        </div>

        <div className="metrics-grid">
          <Metric label="Messages" value={formatNumber(stats?.totalMessages)} />
          <Metric label="Sent" value={formatNumber(stats?.sentCount)} />
          <Metric label="Received" value={formatNumber(stats?.receivedCount)} />
          <Metric label="Active days" value={formatNumber(stats?.activeDays)} />
        </div>

        <DynamicsRow dynamics={preview?.dynamics} lastTimestamp={stats?.lastTimestamp} />

        <div className="cost-row">
          <div>
            <span>Estimated cost</span>
            <strong>{formatMoney(preview?.estimate?.estimated_cost)}</strong>
            <p>
              {formatNumber(preview?.estimate?.input_tokens)} input tokens /{' '}
              {formatNumber(preview?.estimate?.output_tokens)} output tokens
            </p>
          </div>
          <button
            type="button"
            className="button primary"
            disabled={!canRun || jobBusy}
            onClick={() => {
              void onRunReport()
            }}
          >
            <SparkIcon className="button-icon" />
            {jobBusy ? 'Running...' : 'Run report'}
          </button>
        </div>
      </div>

      {runningJob ? (
        <div className={`job-banner ${runningJob.status}`}>
          <strong>{runningJob.status === 'failed' ? 'Report failed' : `Report ${runningJob.status}`}</strong>
          <span>{runningJob.error || lastLog(runningJob.logs) || 'Working through the selected conversation.'}</span>
        </div>
      ) : null}

      <div className="chart-block">
        <div className="block-heading">
          <div>
            <h2>Monthly activity</h2>
            <p>Messages by month.</p>
          </div>
          <span>{formatNumber(preview?.monthly?.length || 0)} months</span>
        </div>
        <ActivityChart data={preview?.monthly || []} />
      </div>

      <div className="recent-block">
        <div className="block-heading">
          <div>
            <h2>Recent messages</h2>
          </div>
          <span>Latest</span>
        </div>
        <div className="message-list">
          {(preview?.recentMessages || []).map((message) => (
            <div key={`${message.timestamp}-${message.text}`} className="message-row">
              <span>{shortDate(message.timestamp)}</span>
              <p className={isOutbound(message) ? 'outbound' : ''}>{message.text || '(empty)'}</p>
            </div>
          ))}
          {!preview?.recentMessages?.length ? (
            <div className="empty-state">
              <strong>No preview loaded.</strong>
              <span>Select a conversation to see recent messages.</span>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function lastLog(logs: Array<string | { message?: string }> = []) {
  for (let index = logs.length - 1; index >= 0; index -= 1) {
    const entry = logs[index]
    const message = typeof entry === 'string' ? entry : entry?.message || ''
    if (message && !message.startsWith('127.0.0.1 - -')) return message
  }
  return ''
}

function DynamicsRow({ dynamics, lastTimestamp }: { dynamics?: Dynamics; lastTimestamp?: string }) {
  const [renderedAt] = useState(() => Date.now())
  if (!dynamics) return null
  const pct = (value?: number | null) =>
    typeof value === 'number' ? `${Math.round(value * 100)}%` : null
  const balance = pct(dynamics.balanceRecent ?? dynamics.balanceLifetime)
  const balanceLifetime = pct(dynamics.balanceLifetime)
  const initiation = pct(dynamics.initiationRecent ?? dynamics.initiationLifetime)
  const trend = dynamics.volumeTrendPct
  const speakers = dynamics.topSpeakers ?? []
  // derive the quiet streak live -- the cached payload's value freezes and
  // drifts a day behind per day
  const lastMs = lastTimestamp ? Date.parse(lastTimestamp) : NaN
  const quietDays = Number.isFinite(lastMs)
    ? Math.max(0, Math.floor((renderedAt - lastMs) / 86_400_000))
    : dynamics.quietDays

  const hasAnything = balance || initiation || typeof trend === 'number' || speakers.length
  if (!hasAnything) return null

  return (
    <div className="dynamics-block">
      <span className="dynamics-label">Dynamics</span>
      <div className="dynamics-grid">
        {balance ? (
          <div className="dynamic-stat">
            <span>You carry</span>
            <strong>{balance}</strong>
            <small>
              {balanceLifetime && balanceLifetime !== balance
                ? `of recent messages · ${balanceLifetime} lifetime`
                : 'of the messages'}
            </small>
          </div>
        ) : null}
        {initiation ? (
          <div className="dynamic-stat">
            <span>You open the day</span>
            <strong>{initiation}</strong>
            <small>of active days, last 90d</small>
          </div>
        ) : null}
        {typeof trend === 'number' ? (
          <div className="dynamic-stat">
            <span>Volume trend</span>
            <strong className={trend < 0 ? 'down' : 'up'}>
              {trend > 0 ? '+' : ''}
              {trend}%
            </strong>
            <small>last 3 months vs lifetime</small>
          </div>
        ) : null}
        {typeof quietDays === 'number' && quietDays > 30 ? (
          <div className="dynamic-stat">
            <span>Quiet for</span>
            <strong>{formatNumber(quietDays)}d</strong>
            <small>since the last message</small>
          </div>
        ) : null}
      </div>
      {speakers.length ? (
        <div className="dynamics-speakers">
          {speakers.map((speaker) => (
            <span key={speaker.name} className="speaker-chip">
              {speaker.name} <em>{Math.round(speaker.share * 100)}%</em>
            </span>
          ))}
        </div>
      ) : null}
    </div>
  )
}
