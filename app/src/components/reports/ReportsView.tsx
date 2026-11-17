import { useMemo, useState, type CSSProperties } from 'react'
import { LIST_WIDTH } from '../../lib/layout'
import { reportEntries, type ReportEntry } from '../../lib/reports'
import type { AnalysisPayload, ReportFile } from '../../types'
import { ChevronLeftIcon } from '../Icons'
import { ResizeHandle } from '../ResizeHandle'
import { ReportList } from './ReportList'
import { ReportPage, ReportPageSkeleton } from './ReportPage'

type ReportsViewProps = {
  reports: ReportFile[]
  selectedReport: ReportFile | null
  // the opened report, or null while it loads or when it could not be opened
  analysis: AnalysisPayload | null
  loading: boolean
  // the chosen report could not be opened
  failed: boolean
  listWidth: number
  onListWidthChange: (value: number) => void
  onSelectReport: (report: ReportFile) => void
  onOpenConversation: (chatId: string, name: string) => void
  // to where a first report is made
  onChooseConversation: () => void
}

// Saved reports on the left, the chosen one on the right as a page to read.
export function ReportsView({
  reports,
  selectedReport,
  analysis,
  loading,
  failed,
  listWidth,
  onListWidthChange,
  onSelectReport,
  onOpenConversation,
  onChooseConversation,
}: ReportsViewProps) {
  // in a narrow window only one of the two panes shows
  const [pane, setPane] = useState<'list' | 'detail'>('detail')
  const entries = useMemo(() => reportEntries(reports), [reports])
  const selected = entries.find((entry) => entry.files.some((file) => file.path === selectedReport?.path)) ?? null
  // a report that is chosen and not yet opened is on its way
  const waiting = loading || Boolean(selected && !analysis && !failed)

  if (!entries.length) {
    return (
      <div className="view">
        <div className="page arrive-each">
          <div className="empty">
            <p>No reports yet</p>
            <button type="button" className="btn" onClick={onChooseConversation}>
              Analyze a conversation
            </button>
          </div>
        </div>
      </div>
    )
  }

  function open(entry: ReportEntry) {
    onSelectReport(entry.report)
    setPane('detail')
  }

  return (
    <div className="split" data-pane={pane} style={{ '--list-w': `${listWidth}px` } as CSSProperties}>
      <ReportList entries={entries} selectedKey={selected?.key ?? null} onSelect={open} />
      <ResizeHandle
        label="Resize report list"
        value={listWidth}
        min={LIST_WIDTH.min}
        max={LIST_WIDTH.max}
        onChange={onListWidthChange}
      />
      <section
        // a new report starts at the top
        key={selected?.key}
        className="pane pane-detail"
        aria-label={selected?.title ?? 'Report'}
        aria-busy={waiting}
      >
        <div className="detail">
          <button type="button" className="pane-back btn quiet" onClick={() => setPane('list')}>
            <ChevronLeftIcon />
            Reports
          </button>
          {waiting ? (
            <ReportPageSkeleton />
          ) : selected && analysis ? (
            <ReportPage
              entry={selected}
              analysis={analysis}
              onOpenConversation={() => onOpenConversation(analysis.contact, selected.title)}
            />
          ) : selected ? (
            <div className="empty">
              <p>This report could not be opened</p>
              <button type="button" className="btn" onClick={() => onSelectReport(selected.report)}>
                Try again
              </button>
            </div>
          ) : null}
        </div>
      </section>
    </div>
  )
}
