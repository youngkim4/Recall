import { useState, type KeyboardEvent, type PointerEvent } from 'react'
import { roundUp, type AxisLabel } from '../../lib/activity'
import { formatNumber } from '../../lib/format'
import type { StatsRow } from '../../lib/insights'
import { useTweened } from '../../lib/useTweened'

type ColumnChartProps = {
  rows: StatsRow[]
  // the names under the columns, by the index of the column they stand under
  labels: AxisLabel[]
  // the name of one column when it is read: "March 2025", "Monday"
  nameOf: (row: StatsRow) => string
  // what the chart shows, for a reader that cannot see it
  summary: string
}

const compact = new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 })

// The answer to a question of Insights as columns, after Amplitude's and
// Pipedrive's charts: one column a period, a weekday or an hour, what you
// sent at its foot and what they sent on top, against a round number at the
// top of the scale. The columns are drawn in one SVG with no text in it, so
// it can stretch to any width; the names stand under it in the page's type.
// Moving across it, by pointer or arrow keys, reads one column: its name
// and its two counts stand over the chart while it is read.
export function ColumnChart({ rows, labels, nameOf, summary }: ColumnChartProps) {
  const [active, setActive] = useState<number | null>(null)
  const count = rows.length
  const top = roundUp(Math.max(1, ...rows.map((row) => row.total)))
  // A new answer does not jump in: each column grows or shrinks from where it
  // stands, in hundredths of the scale, the scale with it.
  const you = useTweened(rows.map((row) => (row.sent / top) * 100))
  const them = useTweened(rows.map((row) => (row.received / top) * 100))
  // while the columns move they can be more or fewer than the answer's
  const drawn = Math.min(you.length, them.length)
  // a gap between columns while they are wide enough to have one
  const gap = drawn > 200 ? 0 : drawn > 60 ? 0.15 : 0.25
  const picked = active === null ? null : rows[active]

  function columnAt(event: PointerEvent<HTMLDivElement>) {
    const box = event.currentTarget.getBoundingClientRect()
    return Math.max(0, Math.min(count - 1, Math.floor(((event.clientX - box.left) / box.width) * count)))
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const steps: Record<string, number> = { ArrowLeft: -1, ArrowRight: 1, Home: -count, End: count }
    if (event.key === 'Escape') return setActive(null)
    if (!(event.key in steps)) return
    event.preventDefault()
    const from = active ?? (steps[event.key] > 0 ? -1 : count)
    setActive(Math.max(0, Math.min(count - 1, from + steps[event.key])))
  }

  return (
    <div className="columns">
      <p className="columns-readout" aria-live="polite">
        {picked ? (
          <>
            <span className="columns-when">{nameOf(picked)}</span>
            <span className="readout-voice">
              <i className="you" />
              You <b>{formatNumber(picked.sent)}</b>
            </span>
            <span className="readout-voice">
              <i className="them" />
              Them <b>{formatNumber(picked.received)}</b>
            </span>
          </>
        ) : null}
      </p>
      <div
        className="columns-plot"
        role="img"
        aria-label={`${summary} Use the arrow keys to read each column.`}
        tabIndex={0}
        // a click reads a column; it does not take the focus, so no ring is drawn
        onMouseDown={(event) => event.preventDefault()}
        data-reading={active === null ? undefined : ''}
        onPointerMove={(event) => setActive(columnAt(event))}
        onPointerLeave={() => setActive(null)}
        onKeyDown={handleKeyDown}
        onBlur={() => setActive(null)}
      >
        {/* the top of the scale: a hairline, its number at the end of it */}
        <span className="columns-top" aria-hidden="true">
          {compact.format(top)}
        </span>
        <svg viewBox={`0 0 ${Math.max(1, drawn)} 100`} preserveAspectRatio="none" aria-hidden="true">
          {Array.from({ length: drawn }, (_, index) => {
            const x = index + gap
            const width = 1 - gap * 2
            return (
              <g key={index} data-active={index === active || undefined}>
                <rect className="you" x={x} y={100 - you[index]} width={width} height={you[index]} />
                <rect className="them" x={x} y={100 - you[index] - them[index]} width={width} height={them[index]} />
              </g>
            )
          })}
        </svg>
      </div>
      <div className="columns-axis" aria-hidden="true">
        {labels.map((label) => (
          <span
            key={label.index}
            data-edge={label.atEdge || undefined}
            style={{ left: `${(label.atEdge ? label.index : label.index + 0.5) * (100 / Math.max(1, count))}%` }}
          >
            {label.text}
          </span>
        ))}
      </div>
    </div>
  )
}
