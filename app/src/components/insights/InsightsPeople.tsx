import type { CSSProperties } from 'react'
import { formatNumber, formatPercent } from '../../lib/format'
import type { StatsPayload } from '../../lib/insights'
import { useRowsThatFit } from '../../lib/useRowsThatFit'
import { Truncate } from '../ui/Truncate'
import { VoiceBar, VoiceLegend } from '../ui/VoiceBar'

export type PeopleBy = 'person' | 'conversation'

type InsightsPeopleProps = {
  by: PeopleBy
  onByChange: (by: PeopleBy) => void
  stats: StatsPayload | null
  // a row opens its conversation (the busiest one of a person)
  onOpen: (conversationId: string, label: string) => void
}

// Alone in a column the list shows eight rows. Beside the hours of the week,
// in a wide window, it shows as many as fit there, so the two end on one
// line, and never fewer than five (useRowsThatFit, as the Overview's list
// beside the moments). The server is asked for as many as can ever show.
const SHOWN = 8
const FEWEST = 5
export const PEOPLE_ASKED = 14
// a row is 12px above and below a 20px line, with a hairline over it
const rowHeight = (rem: number) => 2.75 * rem + 1

const TABS: Array<{ by: PeopleBy; label: string }> = [
  { by: 'person', label: 'People' },
  { by: 'conversation', label: 'Conversations' },
]

// Who the counted messages are with, ranked, after Plausible's lists: the
// heading is a pair of tabs (People, Conversations), each row a name, a bar
// against the busiest split into the two voices, the count and its share.
export function InsightsPeople({ by, onByChange, stats, onOpen }: InsightsPeopleProps) {
  const [box, shown] = useRowsThatFit(rowHeight, SHOWN, FEWEST)
  const rows = (stats?.rows ?? []).slice(0, shown)
  const largest = Math.max(1, rows[0]?.total ?? 1)
  const total = Math.max(1, stats?.totals.total ?? 1)
  return (
    <section className="section ruled insights-people" aria-label={by === 'person' ? 'People' : 'Conversations'}>
      <div className="section-head">
        <div className="section-tabs" role="tablist" aria-label="Count by">
          {TABS.map((tab) => (
            <button
              key={tab.by}
              type="button"
              role="tab"
              aria-selected={tab.by === by}
              className="section-tab"
              onClick={() => onByChange(tab.by)}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <VoiceLegend />
      </div>
      <div ref={box} className="ranked-list">
        <ol
          className="rows ruled"
          role="list"
          aria-busy={!stats || undefined}
          style={{ '--rows': shown } as CSSProperties}
        >
          {stats && !rows.length ? <li className="pane-note">No one in these messages</li> : null}
          {(stats ? rows : Array.from({ length: shown }, () => null)).map((row, index) =>
            row ? (
              <li key={row.key}>
                <button
                  type="button"
                  className="row ranked-row"
                  aria-label={`${row.label}: ${formatNumber(row.total)} messages, ${formatPercent(row.total / total)}`}
                  onClick={() => onOpen(row.key, row.label)}
                >
                  <span className="ranked-place">{index + 1}</span>
                  <Truncate className="row-title">{row.label}</Truncate>
                  <span className="ranked-bar">
                    <VoiceBar sent={row.sent} received={row.received} scale={row.total / largest} />
                  </span>
                  <span className="ranked-count">{formatNumber(row.total)}</span>
                  <span className="ranked-share">{formatPercent(row.total / total)}</span>
                </button>
              </li>
            ) : (
              <li key={index}>
                <div className="row ranked-row" aria-hidden="true">
                  <span className="ranked-place">{index + 1}</span>
                  <span>
                    <span className="skeleton">Conversation name</span>
                  </span>
                </div>
              </li>
            ),
          )}
        </ol>
      </div>
    </section>
  )
}
