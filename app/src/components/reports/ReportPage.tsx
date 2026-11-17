import { useEffect, useId, useRef } from 'react'
import { recallBridge } from '../../lib/bridge'
import { conversationName, isGroupChat } from '../../lib/conversations'
import { formatDate, formatDateRange } from '../../lib/format'
import type { ReportEntry } from '../../lib/reports'
import type { AnalysisPayload } from '../../types'
import { VoicesOverTime } from '../ui/VoicesOverTime'
import { ReportFigures } from './ReportFigures'
import { ReportMoments } from './ReportMoments'
import { ReportNumbers } from './ReportNumbers'
import { ReportStory } from './ReportStory'
import { usePaper } from './usePaper'

type ReportPageProps = {
  entry: ReportEntry
  analysis: AnalysisPayload
  onOpenConversation: () => void
}

const FILE_NAMES: Record<string, string> = { md: 'Markdown', html: 'Web page', csv: 'Table of moments' }

// A saved report, set as a document. What the numbers say comes first: who
// and when, the figures, the two voices over time, the smaller figures. Then
// the moments it picked out, as a timeline; then the story that was written,
// in chapters. Export sets the same document on paper.
export function ReportPage({ entry, analysis, onOpenConversation }: ReportPageProps) {
  const title = conversationName(analysis.contactDisplayName || entry.report.displayName, analysis.contact)
  const group = isGroupChat(analysis.contact)
  const { stats, activity } = analysis
  const baseId = useId()
  const { paper, open, close, print } = usePaper(`Recall report, ${title}`)
  const them = group ? 'Everyone else' : title

  const exporter = useRef<HTMLButtonElement>(null)

  // Escape leaves the paper, as it leaves anything that took over the window.
  // The button that opened it is hidden meanwhile, so the focus is handed
  // back to it on the way out.
  useEffect(() => {
    if (!paper) return
    const button = exporter.current
    const leave = (event: KeyboardEvent) => {
      if (event.key === 'Escape') close()
    }
    window.addEventListener('keydown', leave)
    return () => {
      window.removeEventListener('keydown', leave)
      button?.focus()
    }
  }, [paper, close])

  return (
    <article className="report-page arrive-each">
      {paper ? (
        <div className="paper-bar" data-screen-only>
          <button type="button" className="btn quiet" onClick={close}>
            Done
          </button>
          {/* the Mac app saves the file itself; a browser shows its print sheet, which can save one */}
          <button type="button" className="btn primary" autoFocus onClick={print}>
            {recallBridge.available ? 'Save as PDF' : 'Print or save as PDF'}
          </button>
        </div>
      ) : null}

      <header className="report-cover">
        <p className="report-mark" data-paper-only>
          Recall
        </p>
        <div className="section-head">
          <h2 className="detail-title">{title}</h2>
          <span className="section-links" data-screen-only>
            <button type="button" className="link" onClick={onOpenConversation}>
              Open conversation
            </button>
            <button ref={exporter} type="button" className="btn" onClick={open}>
              Export PDF
            </button>
          </span>
        </div>
        <p className="detail-sub">
          {formatDateRange(stats.firstTimestamp, stats.lastTimestamp)}. Written {formatDate(entry.writtenAt)}.
        </p>
      </header>

      <ReportFigures stats={stats} />

      {/* the two voices over time and the smaller figures: one under the other
          in a narrow pane, fitted together side by side in a wide one */}
      <div className="section report-data">
        <section className="report-voices" aria-labelledby={`${baseId}-voices`}>
          <div className="section-head">
            <h3 id={`${baseId}-voices`}>By {activity.unit}</h3>
          </div>
          <div className="card card-pad">
            <VoicesOverTime
              // a new stretch of time starts with nothing picked out
              key={`${activity.unit} ${activity.points[0]?.key} ${activity.points.length}`}
              points={activity.points}
              unit={activity.unit}
              sent={stats.sentCount ?? 0}
              received={stats.receivedCount ?? 0}
              them={them}
            />
          </div>
        </section>
        <ReportNumbers analysis={analysis} them={them} group={group} />
      </div>

      <ReportMoments
        events={analysis.events ?? []}
        first={(stats.firstTimestamp ?? '').slice(0, 10)}
        last={(stats.lastTimestamp ?? '').slice(0, 10)}
        paper={paper}
      />

      {analysis.shownPlain ? (
        <p className="section section-note" data-screen-only>
          This story is long, so it is shown as plain paragraphs.
        </p>
      ) : null}
      <ReportStory html={analysis.summaryHtml || ''} />

      {(analysis.files ?? []).length ? (
        <footer className="section report-files" data-screen-only>
          <span>Saved as</span>
          <span>
            {(analysis.files ?? []).map((file) => (
              <a
                key={file.path}
                className="link"
                href={`/api/report?path=${encodeURIComponent(file.path)}`}
                target="_blank"
                rel="noreferrer"
              >
                {FILE_NAMES[file.kind] ?? file.name}
              </a>
            ))}
          </span>
        </footer>
      ) : null}
    </article>
  )
}

// holds the place of a report while it is read from disk
export function ReportPageSkeleton() {
  return (
    <div className="report-page" aria-hidden="true">
      <p className="detail-title">
        <span className="skeleton">Name of a conversation</span>
      </p>
      <p className="detail-sub">
        <span className="skeleton">Jan 1 – Dec 31, 2024. Written Jan 1, 2025.</span>
      </p>
      <div className="report-figures">
        {['00,000', '000', '0,000', '00 days'].map((value) => (
          <div key={value} className="report-figure card">
            <p>
              <span className="skeleton">Name of it</span>
            </p>
            <p>
              <span className="skeleton">{value}</span>
            </p>
          </div>
        ))}
      </div>
      <div className="section report-data">
        <div className="report-voices">
          <div className="card card-pad">
            <div className="skeleton report-skeleton-chart" />
          </div>
        </div>
        {/* in a wide pane the smaller figures have their places beside the chart */}
        <div className="section report-skeleton-numbers">
          <div className="report-numbers">
            {[0, 1, 2].map((place) => (
              <div key={place} className="report-number card card-pad">
                <p>
                  <span className="skeleton">Name of it</span>
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="section prose">
        {[92, 100, 96, 100, 64].map((width, index) => (
          <p key={index}>
            <span className="skeleton report-skeleton-line" style={{ width: `${width}%` }} />
          </p>
        ))}
      </div>
    </div>
  )
}
