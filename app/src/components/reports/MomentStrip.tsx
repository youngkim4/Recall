import { useMemo, type CSSProperties, type KeyboardEvent } from 'react'
import { formatDate, formatNumber, plural } from '../../lib/format'
import { placeOnSpan, spanTicks, type Moment } from '../../lib/reportMoments'

type MomentStripProps = {
  // the moments that have a day, oldest first
  moments: Moment[]
  // the first and last day the report covers
  first: string
  last: string
  // the keys of the ones that matter most
  leading: Set<string>
  active: string | null
  onActive: (key: string | null) => void
  onPick: (key: string) => void
}

const NEAR_END = 0.14

// The moments along the days the report covers: a mark where each one fell,
// larger for the ones that matter most. Moving across it, by pointer or arrow
// keys, reads a moment; a press or Return goes to it in the list below.
export function MomentStrip({ moments, first, last, leading, active, onActive, onPick }: MomentStripProps) {
  const placed = useMemo(
    () =>
      moments
        .map((moment) => ({ moment, at: placeOnSpan(moment.day, first, last) }))
        .filter((entry): entry is { moment: Moment; at: number } => entry.at !== null),
    [moments, first, last],
  )
  // a mark this near an end would sit under the day that is written there
  const ticks = useMemo(
    () => spanTicks(first, last).filter((tick) => tick.at > NEAR_END && tick.at < 1 - NEAR_END),
    [first, last],
  )
  if (!placed.length) return null
  const activeAt = placed.findIndex((entry) => entry.moment.key === active)

  // the moment whose mark is closest to where the pointer is
  function nearest(event: { currentTarget: HTMLDivElement; clientX: number }) {
    const box = event.currentTarget.getBoundingClientRect()
    const at = (event.clientX - box.left) / box.width
    return placed.reduce((best, entry) => (Math.abs(entry.at - at) < Math.abs(best.at - at) ? entry : best)).moment.key
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === 'Escape') {
      onActive(null)
      return
    }
    if ((event.key === 'Enter' || event.key === ' ') && active) {
      event.preventDefault()
      onPick(active)
      return
    }
    const step = event.key === 'ArrowLeft' ? -1 : event.key === 'ArrowRight' ? 1 : 0
    if (!step) return
    event.preventDefault()
    // the first press of an arrow lands on the newest moment
    const to = activeAt < 0 ? placed.length - 1 : Math.max(0, Math.min(placed.length - 1, activeAt + step))
    onActive(placed[to].moment.key)
  }

  const read = activeAt < 0 ? null : placed[activeAt].moment
  const marked = placed.filter((entry) => leading.has(entry.moment.key)).length

  return (
    <div className="report-strip">
      {/* one line that is always there: what the marks mean, or the moment being read */}
      <p className="report-strip-readout">
        {read ? (
          <>
            <span>{read.title || 'Untitled moment'}</span>
            <b>{formatDate(read.day)}</b>
          </>
        ) : (
          <>
            {marked ? (
              <span className="report-strip-key" data-lead>
                {formatNumber(marked)} that matter most
              </span>
            ) : null}
            {placed.length > marked ? (
              <span className="report-strip-key">{plural(placed.length - marked, 'other')}</span>
            ) : null}
          </>
        )}
      </p>
      <p className="sr-only" aria-live="polite">
        {read ? `${formatDate(read.day)}: ${read.title || 'Untitled moment'}` : ''}
      </p>
      <div
        className="report-strip-line"
        tabIndex={0}
        role="group"
        aria-label={`${placed.length} moments from ${formatDate(first)} to ${formatDate(last)}. Use the arrow keys to read one, and Return to go to it.`}
        onPointerMove={(event) => {
          const key = nearest(event)
          if (key !== active) onActive(key)
        }}
        onPointerLeave={() => onActive(null)}
        onClick={(event) => onPick(nearest(event))}
        onKeyDown={handleKeyDown}
        onBlur={() => onActive(null)}
      >
        {placed.map(({ moment, at }) => (
          <i
            key={moment.key}
            data-lead={leading.has(moment.key) || undefined}
            data-active={moment.key === active || undefined}
            style={{ '--at': at } as CSSProperties}
          />
        ))}
      </div>
      <div className="report-strip-axis" aria-hidden="true">
        <span>{formatDate(first)}</span>
        {ticks.map((tick) => (
          <span key={`${tick.label}-${tick.at}`} className="report-strip-tick" style={{ '--at': tick.at } as CSSProperties}>
            {tick.label}
          </span>
        ))}
        <span>{formatDate(last)}</span>
      </div>
    </div>
  )
}
