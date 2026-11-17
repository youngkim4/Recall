import { useMemo } from 'react'
import { contactTitle, conversationName, isGroupChat, isOutbound, messageCount } from '../../lib/conversations'
import { formatDateRange, plural } from '../../lib/format'
import type { Contact, PreviewPayload, ReportFile } from '../../types'
import { ChevronLeftIcon } from '../Icons'
import type { ThreadMessage } from '../ui/Thread'
import { Figures } from './Figures'
import { FiguresSkeleton } from './FiguresSkeleton'
import { LatestMessages } from './LatestMessages'
import { ReportBar } from './ReportBar'

type Scope = { since: string; until: string }

const NO_DAYS = { start: '', counts: [] }

type ConversationDetailProps = {
  contact: Contact
  // null while it is first worked out
  preview: PreviewPayload | null
  // false while the figures are from before a change of dates or quality
  settled: boolean
  // the figures that were asked for could not be worked out
  failed: boolean
  onRetry: () => void
  reports: ReportFile[]
  models: string[]
  model: string
  onModelChange: (model: string) => void
  scope: Scope
  onScopeChange: (scope: Scope) => void
  canRun: boolean
  running: boolean
  failure?: string
  onRunReport: () => void
  onOpenReport: (report: ReportFile) => void
  // back to the list, where a narrow window shows only one pane
  onShowList: () => void
}

export function ConversationDetail({
  contact,
  preview,
  settled,
  failed,
  onRetry,
  reports,
  models,
  model,
  onModelChange,
  scope,
  onScopeChange,
  canRun,
  running,
  failure,
  onRunReport,
  onOpenReport,
  onShowList,
}: ConversationDetailProps) {
  const title = contactTitle(contact)
  const group = isGroupChat(contact.chat_id)
  const stats = preview?.stats
  const total = stats?.totalMessages ?? messageCount(contact)
  const first = stats?.firstTimestamp ?? contact.first_msg
  const last = stats?.lastTimestamp ?? contact.last_msg

  // by the conversation itself: two people saved under one name each keep their own report
  const latestReport = useMemo(
    () =>
      reports
        .filter((report) => report.kind === 'md' && report.contact === contact.chat_id)
        .sort((a, b) => String(b.updatedAt || '').localeCompare(String(a.updatedAt || '')))[0],
    [reports, contact.chat_id],
  )

  const thread = useMemo<ThreadMessage[]>(
    () =>
      (preview?.recentMessages ?? []).map((message, index) => ({
        key: `${message.timestamp}-${index}`,
        timestamp: String(message.timestamp),
        text: message.text,
        attachment: message.attachment || undefined,
        fromMe: message.isFromMe === null || message.isFromMe === undefined ? null : isOutbound(message),
        sender: group
          ? message.senderName || (message.sender ? conversationName('', message.sender) : undefined)
          : undefined,
      })),
    [preview, group],
  )

  return (
    <section className="pane pane-detail" aria-label={title}>
      <div className="detail analyze-detail arrive-each">
        <button type="button" className="pane-back btn quiet" onClick={onShowList}>
          <ChevronLeftIcon />
          Conversations
        </button>
        <header>
          <div className="section-head">
            <h2 className="detail-title">{title}</h2>
            {latestReport ? (
              <span className="section-links">
                <button type="button" className="link" onClick={() => onOpenReport(latestReport)}>
                  Open latest report
                </button>
              </span>
            ) : null}
          </div>
          <p className="detail-sub">
            {plural(total, 'message')}, {formatDateRange(first, last)}
          </p>
        </header>

        <ReportBar
          models={models}
          model={model}
          onModelChange={onModelChange}
          scope={scope}
          onScopeChange={onScopeChange}
          daily={preview?.daily ?? NO_DAYS}
          estimate={settled ? (preview?.estimate ?? null) : null}
          estimateError={settled ? preview?.estimateError : undefined}
          canRun={canRun}
          running={running}
          failure={failure}
          onRun={onRunReport}
        />

        {failed ? (
          <p className="section section-note" role="alert">
            {preview ? 'These dates could not be read.' : 'This conversation could not be read.'}{' '}
            <button type="button" className="link in-text" onClick={onRetry}>
              Try again
            </button>
          </p>
        ) : null}

        {preview && stats ? (
          <div className="detail-body">
            <Figures figures={preview} title={title} group={group} />

            {thread.length ? <LatestMessages messages={thread} /> : null}
          </div>
        ) : failed ? null : (
          <div className="detail-body" aria-hidden="true">
            <FiguresSkeleton group={group} />
          </div>
        )}
      </div>
    </section>
  )
}
