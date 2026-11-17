import { useMemo, type CSSProperties, type KeyboardEvent, type PointerEvent } from 'react'
import { axisLabels, roundUp, type ActivityPoint, type ActivityUnit } from '../../lib/activity'

type ActivityChartProps = {
  // one bar a day or a month, oldest first
  points: ActivityPoint[]
  unit: ActivityUnit
  // the bar under the pointer or the arrow keys, by index; null for none
  active: number | null
  onActiveChange: (index: number | null) => void
  // read to assistive technology in place of the picture
  summary: string
}

const compact = new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 })
// more labels than this do not fit a narrow chart side by side
const FEW_LABELS = 6

// Messages over time in two voices: what you sent rises above the line in
// blue, what you received hangs below it in gray, on one scale. Moving across
// it, by pointer or arrow keys, picks out a day or a month.
export function ActivityChart({ points, unit, active, onActiveChange, summary }: ActivityChartProps) {
  const { above, below } = useMemo(
    () => ({
      above: roundUp(Math.max(0, ...points.map((point) => point.sent))),
      below: roundUp(Math.max(0, ...points.map((point) => point.received))),
    }),
    [points],
  )
  const labels = useMemo(() => axisLabels(points, unit), [points, unit])
  if (!points.length || above + below === 0) return null

  function pointAt(event: PointerEvent<HTMLDivElement>) {
    const box = event.currentTarget.getBoundingClientRect()
    const position = (event.clientX - box.left) / box.width
    return Math.max(0, Math.min(points.length - 1, Math.floor(position * points.length)))
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === 'Escape') {
      onActiveChange(null)
      return
    }
    const last = points.length - 1
    const step = event.key === 'ArrowLeft' ? -1 : event.key === 'ArrowRight' ? 1 : 0
    // the first press of an arrow lands on the newest bar
    const to = step
      ? active === null
        ? last
        : Math.max(0, Math.min(last, active + step))
      : event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? last
          : null
    if (to === null) return
    event.preventDefault()
    onActiveChange(to)
  }

  return (
    <figure className="activity">
      <figcaption className="sr-only">{summary}</figcaption>
      <div className="activity-frame">
        <div
          className="activity-plot"
          style={{ '--count': points.length, '--above': above, '--below': below } as CSSProperties}
          tabIndex={0}
          role="group"
          aria-label={`Messages by ${unit}. Use the arrow keys to read a ${unit}.`}
          onPointerMove={(event) => onActiveChange(pointAt(event))}
          onPointerLeave={() => onActiveChange(null)}
          onKeyDown={handleKeyDown}
          onBlur={() => onActiveChange(null)}
        >
          {points.map((point, index) => (
            <div key={point.key} className={index === active ? 'activity-slot is-active' : 'activity-slot'}>
              <span className="activity-up">
                <i
                  className="you"
                  style={{ height: `${above ? (point.sent / above) * 100 : 0}%`, minHeight: point.sent ? 1 : 0 }}
                />
              </span>
              <span className="activity-down">
                <i
                  className="them"
                  style={{ height: `${below ? (point.received / below) * 100 : 0}%`, minHeight: point.received ? 1 : 0 }}
                />
              </span>
            </div>
          ))}
        </div>
        {/* the scale: what the top of each side stands for */}
        <div className="activity-scale" aria-hidden="true">
          <span>{above ? compact.format(above) : ''}</span>
          <span>{below ? compact.format(below) : ''}</span>
        </div>
      </div>
      <div className={labels.length > FEW_LABELS ? 'activity-axis is-full' : 'activity-axis'} aria-hidden="true">
        {labels.map((label) => (
          <span
            key={label.index}
            className={label.atEdge ? 'is-first' : undefined}
            style={{ left: label.atEdge ? 0 : `${((label.index + 0.5) / points.length) * 100}%` }}
          >
            {label.text}
          </span>
        ))}
      </div>
    </figure>
  )
}
