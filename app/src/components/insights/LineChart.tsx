import { useState, type KeyboardEvent, type PointerEvent } from 'react'
import { roundUp, type AxisLabel } from '../../lib/activity'
import { formatNumber } from '../../lib/format'
import type { StatsRow } from '../../lib/insights'

export type LineSeries = 'both' | 'sent' | 'received'

type LineChartProps = {
  rows: StatsRow[]
  // which lines show: yours and theirs, or one of them (the tab above the chart)
  series: LineSeries
  labels: AxisLabel[]
  nameOf: (row: StatsRow) => string
  summary: string
}

const compact = new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 })
// the drawing's own units; it stretches to the width it is given, its lines keep their 2px
const WIDE = 1000
const TALL = 100

// The answer over time as lines, after ElevenLabs' usage chart: what you
// sent and what they sent, 2px each, over three quiet gridlines, with the
// top of the scale named at its end. Moving across it, by pointer or arrow
// keys, sets a crosshair on one point and reads it over the chart. A click
// does not focus it, so no ring is drawn for the pointer; the keyboard's
// focus is ringed.
export function LineChart({ rows, series, labels, nameOf, summary }: LineChartProps) {
  const [active, setActive] = useState<number | null>(null)
  const count = rows.length
  const showSent = series !== 'received'
  const showReceived = series !== 'sent'
  const peak = Math.max(1, ...rows.map((row) => Math.max(showSent ? row.sent : 0, showReceived ? row.received : 0)))
  const top = roundUp(peak)
  const x = (index: number) => (count < 2 ? WIDE / 2 : (index / (count - 1)) * WIDE)
  const y = (value: number) => TALL - (value / top) * TALL
  const path = (pick: (row: StatsRow) => number) =>
    rows.map((row, index) => `${index ? 'L' : 'M'}${x(index).toFixed(2)} ${y(pick(row)).toFixed(2)}`).join(' ')
  const picked = active === null ? null : rows[active]
  // a share of the width or the height, for what stands over the drawing in the page's own type
  const across = (index: number) => `${(x(index) / WIDE) * 100}%`
  const up = (value: number) => `${(value / top) * 100}%`

  function pointAt(event: PointerEvent<HTMLDivElement>) {
    const box = event.currentTarget.getBoundingClientRect()
    const share = (event.clientX - box.left) / box.width
    return Math.max(0, Math.min(count - 1, Math.round(share * (count - 1))))
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
    <div className="lines">
      <p className="lines-readout" aria-live="polite">
        {picked ? (
          <>
            <span className="lines-when">{nameOf(picked)}</span>
            {showSent ? (
              <span className="readout-voice">
                <i className="you" />
                You <b>{formatNumber(picked.sent)}</b>
              </span>
            ) : null}
            {showReceived ? (
              <span className="readout-voice">
                <i className="them" />
                Them <b>{formatNumber(picked.received)}</b>
              </span>
            ) : null}
          </>
        ) : (
          // at rest the line over the chart says whose each line is
          <>
            {showSent ? (
              <span className="readout-voice">
                <i className="you" />
                You
              </span>
            ) : null}
            {showReceived ? (
              <span className="readout-voice">
                <i className="them" />
                Them
              </span>
            ) : null}
          </>
        )}
        <span className="lines-top">{compact.format(top)}</span>
      </p>
      <div
        className="lines-plot"
        role="img"
        aria-label={`${summary} Use the arrow keys to read each point.`}
        tabIndex={0}
        // a click reads a point; it does not take the focus, so no ring is drawn
        onMouseDown={(event) => event.preventDefault()}
        onPointerMove={(event) => setActive(pointAt(event))}
        onPointerLeave={() => setActive(null)}
        onKeyDown={handleKeyDown}
        onBlur={() => setActive(null)}
      >
        <svg viewBox={`0 0 ${WIDE} ${TALL}`} preserveAspectRatio="none" aria-hidden="true">
          {[0.25, 0.5, 0.75].map((share) => (
            <line key={share} className="lines-grid" x1="0" x2={WIDE} y1={TALL * share} y2={TALL * share} />
          ))}
          {showReceived ? <path className="lines-them" d={path((row) => row.received)} /> : null}
          {showSent ? <path className="lines-you" d={path((row) => row.sent)} /> : null}
        </svg>
        {picked && active !== null ? (
          <>
            <span className="lines-cross" style={{ left: across(active) }} aria-hidden="true" />
            {showReceived ? (
              <span className="lines-dot them" style={{ left: across(active), bottom: up(picked.received) }} aria-hidden="true" />
            ) : null}
            {showSent ? (
              <span className="lines-dot you" style={{ left: across(active), bottom: up(picked.sent) }} aria-hidden="true" />
            ) : null}
          </>
        ) : null}
      </div>
      <div className="columns-axis" aria-hidden="true">
        {labels.map((label) => (
          <span key={label.index} data-edge={label.atEdge || undefined} style={{ left: across(label.index) }}>
            {label.text}
          </span>
        ))}
      </div>
    </div>
  )
}
