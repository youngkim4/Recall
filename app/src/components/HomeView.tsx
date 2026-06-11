import { useEffect, useMemo, useState } from 'react'
import type { Contact, Defaults, Job, MemoriesPayload, ReportFile, ViewKey } from '../types'
import { contactTitle, formatNumber, reportTitle, shortDateTime } from '../lib/format'
import { recallApi } from '../lib/api'
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
  onReplayFirstWords: () => void
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
  defaults,
  contacts,
  reports,
  reportCount,
  namedCount,
  onViewChange,
  onSelectContact,
  onOpenReport,
  onReplayFirstWords,
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

  const top = ranked[0]
  const topPct = top && totalMessages ? Math.round((Number(top.message_count || 0) / totalMessages) * 100) : 0
  const topSixTotal = topSix.reduce((sum, contact) => sum + Number(contact.message_count || 0), 0)
  const topSixShare = totalMessages ? Math.round((topSixTotal / totalMessages) * 100) : 0
  const topLabel = top ? contactTitle(top) : '—'
  const overviewRead = top
    ? `Across ${formatNumber(contacts.length)} conversation${contacts.length === 1 ? '' : 's'}${
        namedCount ? `, ${formatNumber(namedCount)} of them named` : ''
      }. ${contactTitle(top)} leads with ${formatNumber(Number(top.message_count || 0))} — about ${topPct}% of everything${
        ranked[1] ? `, then ${contactTitle(ranked[1])}` : ''
      }.`
    : ''

  return (
    <section className="home-view" aria-label="Overview">
      <div className="home-hero ov-read">
        <div className="ov-read-main">
          <span className="report-read-label">Your message library</span>
          <strong className="report-read-number">{formatNumber(totalMessages)}</strong>
          {overviewRead ? <p className="report-read-line">{overviewRead}</p> : null}
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
          <span className="kpi-label">Conversations</span>
          <span className="kpi-value">{formatNumber(contacts.length)}</span>
          <span className="kpi-foot">
            <span className="kpi-sub">in your archive</span>
          </span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Named</span>
          <span className="kpi-value">{formatNumber(namedCount)}</span>
          <span className="kpi-foot">
            <span className="kpi-sub">of {formatNumber(contacts.length)}</span>
          </span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Top conversation</span>
          <span className="kpi-value">{top ? formatNumber(Number(top.message_count || 0)) : '0'}</span>
          <span className="kpi-foot">
            <span className="kpi-sub">{top ? contactTitle(top) : 'No data'}</span>
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

      <MemoriesSection
        messagesPath={defaults?.messagesPath || ''}
        contacts={contacts}
        onSelectContact={onSelectContact}
        onReplayFirstWords={defaults?.hasMessages ? onReplayFirstWords : undefined}
      />

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

        <section className="panel ov-share">
          <div className="panel-head">
            <h3>Share of messages</h3>
          </div>
          {donutSegments.length ? (
            <div className="ov-share-body">
              <Donut
                segments={donutSegments}
                centerLabel={formatNumber(totalMessages)}
                centerSub="messages"
                formatValue={sharePercent}
                size={150}
              />
              <div className="share-insight">
                <div className="share-stat">
                  <span>Most active</span>
                  <strong>{topLabel} · {topPct}%</strong>
                </div>
                <div className="share-stat">
                  <span>Top 6 share</span>
                  <strong>{topSixShare}%</strong>
                </div>
                <div className="share-stat">
                  <span>Conversations</span>
                  <strong>{formatNumber(contacts.length)}</strong>
                </div>
              </div>
            </div>
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

function MemoriesSection({
  messagesPath,
  contacts,
  onSelectContact,
  onReplayFirstWords,
}: {
  messagesPath: string
  contacts: Contact[]
  onSelectContact: (contact: Contact) => void
  onReplayFirstWords?: () => void
}) {
  const [memories, setMemories] = useState<MemoriesPayload | null>(null)

  useEffect(() => {
    if (!messagesPath) return
    let cancelled = false
    recallApi
      .memories({ messagesPath })
      .then((response) => {
        if (!cancelled) setMemories(response.memories)
      })
      .catch(() => {
        // memories are a bonus; the overview works without them
      })
    return () => {
      cancelled = true
    }
  }, [messagesPath])

  const cards = useMemo(() => {
    if (!memories) return []
    const anniversary = memories.anniversaries.slice(0, 2).map((item) => ({
      key: `anni-${item.chatId}`,
      kicker: item.inDays === 0 ? 'Anniversary today' : `Anniversary in ${item.inDays}d`,
      tone: 'anniversary',
      name: item.name,
      line: `${item.years} ${item.years === 1 ? 'year' : 'years'} since your first message (${item.date}).`,
      chatId: item.chatId,
    }))
    const onThisDay = memories.onThisDay.slice(0, 3).map((item) => ({
      key: `otd-${item.chatId}-${item.year}`,
      kicker: `On this day · ${item.year}`,
      tone: 'onthisday',
      name: item.name,
      line: item.preview ? `“${item.preview}”` : `${formatNumber(item.count)} messages that day.`,
      chatId: item.chatId,
    }))
    const reconnect = memories.reconnect.slice(0, 2).map((item) => ({
      key: `rec-${item.chatId}`,
      kicker: 'Been a while',
      tone: 'reconnect',
      name: item.name,
      line: `${formatNumber(item.count)} messages together — quiet since ${item.lastDate}.`,
      chatId: item.chatId,
    }))
    const all = [...anniversary, ...onThisDay, ...reconnect]
    // complete rows only (3-up grid) -- a half-empty row reads as a layout bug
    const fullRows = Math.floor(all.length / 3) * 3
    return all.slice(0, Math.min(6, Math.max(fullRows, Math.min(all.length, 3))))
  }, [memories])

  if (!cards.length && !onReplayFirstWords) return null

  return (
    <section className="panel memories-panel">
      <div className="panel-head">
        <h3>Memories</h3>
      </div>
      <div className="memories-row">
        {onReplayFirstWords ? (
          <button type="button" className="memory-card firstwords" onClick={onReplayFirstWords}>
            <span className="memory-kicker">The beginning</span>
            <strong>First words</strong>
            <p>Relive the first thing you and your favorite people ever said.</p>
          </button>
        ) : null}
        {cards.map((card) => {
          // every card navigates -- contacts outside the top-80 list get a
          // synthetic Contact so the click never silently does nothing
          const contact =
            contacts.find((item) => item.chat_id === card.chatId) ??
            ({ chat_id: card.chatId, display_name: card.name } as Contact)
          return (
            <button
              key={card.key}
              type="button"
              className={`memory-card ${card.tone}`}
              onClick={() => onSelectContact(contact)}
            >
              <span className="memory-kicker">{card.kicker}</span>
              <strong>{card.name}</strong>
              <p>{card.line}</p>
            </button>
          )
        })}
      </div>
    </section>
  )
}
