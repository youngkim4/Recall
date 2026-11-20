import type { CSSProperties } from 'react'
import { formatNumber } from '../../lib/format'
import { statsKey } from '../../lib/insights'
import { loadStats } from '../../lib/insightsLoad'
import { LATELY_DAYS, sparkAsk, type LatelyRow } from '../../lib/lately'
import { LIST_FEWEST, listRowHeight } from '../../lib/overviewFit'
import { useLoad } from '../../lib/useLoad'
import { useRowsThatFit } from '../../lib/useRowsThatFit'
import { Truncate } from '../ui/Truncate'

type LatelyProps = {
  // null while they load; as many as LIST_MOST, of which the rows that fit are shown
  rows: LatelyRow[] | null
  messagesPath: string
  // the last day of the archive: the thirty days end there
  lastDay: string
  onOpen: (row: LatelyRow) => void
  onOpenInsights: () => void
}

const SPARK_W = 96
const SPARK_H = 20

// One person's last eight weeks as a small line, in ink. It stretches to the
// room its column has; the stroke keeps its width however far it stretches.
function Spark({ messagesPath, row, lastDay }: { messagesPath: string; row: LatelyRow; lastDay: string }) {
  const { load } = useLoad(statsKey(messagesPath, sparkAsk(row, lastDay)), loadStats)
  if (load.state !== 'ready' || load.value.rows.length < 2) return <span className="lately-spark" aria-hidden="true" />
  const weeks = load.value.rows.map((week) => week.total)
  const top = Math.max(1, ...weeks)
  const path = weeks
    .map((value, index) => {
      const x = (index / (weeks.length - 1)) * SPARK_W
      const y = SPARK_H - 1 - (value / top) * (SPARK_H - 2)
      return `${index ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`
    })
    .join(' ')
  return (
    <svg className="lately-spark" viewBox={`0 0 ${SPARK_W} ${SPARK_H}`} preserveAspectRatio="none" aria-hidden="true">
      <path d={path} />
    </svg>
  )
}

function Change({ change }: { change: number | null }) {
  if (change === null) return <span className="lately-change">New</span>
  const way = change > 0 ? 'up' : change < 0 ? 'down' : 'level'
  return (
    <span className="lately-change" data-way={way}>
      {change > 0 ? '+' : change < 0 ? '−' : ''}
      {formatNumber(Math.abs(change))}%
    </span>
  )
}

// Who the messages have been with lately, after Spotify's "Recently played"
// row: the people of the archive's last thirty days, the busiest first, each
// with their last eight weeks as a small line, the count, and how it moved
// against the thirty days before, in the theme's green or red. The way to
// Insights is in its heading. It shows as many people as reach the foot of
// the window (overviewFit.ts).
export function Lately({ rows, messagesPath, lastDay, onOpen, onOpenInsights }: LatelyProps) {
  const [box, fit] = useRowsThatFit(listRowHeight, LIST_FEWEST, LIST_FEWEST)
  const shown = rows ? rows.slice(0, fit) : Array.from({ length: fit }, () => null)
  return (
    <section className="section" aria-labelledby="lately-title">
      <div className="section-head">
        <h2 id="lately-title">Lately</h2>
        <span className="meta">Last {LATELY_DAYS} days</span>
        <span className="section-links">
          <button type="button" className="link" onClick={onOpenInsights}>
            Insights
          </button>
        </span>
      </div>
      <div ref={box} className="overview-fit">
        <ol
          className="rows ruled overview-list"
          role="list"
          aria-busy={!rows || undefined}
          style={{ '--rows': Math.max(1, shown.length) } as CSSProperties}
        >
          {shown.map((row, index) =>
            row ? (
              <li key={row.key}>
                <button
                  type="button"
                  className="row lately-row"
                  aria-label={`${row.label}: ${formatNumber(row.total)} messages in the last ${LATELY_DAYS} days`}
                  onClick={() => onOpen(row)}
                >
                  <Truncate className="row-title">{row.label}</Truncate>
                  <Spark messagesPath={messagesPath} row={row} lastDay={lastDay} />
                  <span className="lately-count">{formatNumber(row.total)}</span>
                  <Change change={row.change} />
                </button>
              </li>
            ) : (
              <li key={index}>
                <div className="row lately-row" aria-hidden="true">
                  <span>
                    <span className="skeleton">Conversation name</span>
                  </span>
                </div>
              </li>
            ),
          )}
          {rows && !rows.length ? <li className="pane-note">No messages in the last {LATELY_DAYS} days</li> : null}
        </ol>
      </div>
    </section>
  )
}
