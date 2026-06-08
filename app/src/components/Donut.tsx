export type DonutSegment = {
  key: string
  label: string
  value: number
  color: string
}

type DonutProps = {
  segments: DonutSegment[]
  centerLabel: string
  centerSub?: string
  formatValue?: (value: number) => string
}

const SIZE = 132
const RADIUS = 54
const STROKE = 14
const CENTER = SIZE / 2
const CIRC = 2 * Math.PI * RADIUS

// SVG ring with a center total + legend (Tremor DonutChart)
export function Donut({ segments, centerLabel, centerSub, formatValue }: DonutProps) {
  const sum = segments.reduce((total, seg) => total + seg.value, 0) || 1
  const lengths = segments.map((seg) => (seg.value / sum) * CIRC)
  // cumulative start offset per segment, computed immutably
  const offsets = lengths.map((_, index) => lengths.slice(0, index).reduce((a, b) => a + b, 0))

  return (
    <div className="donut">
      <div className="donut-svg">
        <svg viewBox={`0 0 ${SIZE} ${SIZE}`} role="img" aria-label="Share by segment">
          <circle
            cx={CENTER}
            cy={CENTER}
            r={RADIUS}
            fill="none"
            stroke="var(--surface-3)"
            strokeWidth={STROKE}
          />
          {segments.map((seg, index) => {
            const length = lengths[index]
            const dashArray = `${length} ${CIRC - length}`
            const dashOffset = -offsets[index]
            return (
              <circle
                key={seg.key}
                cx={CENTER}
                cy={CENTER}
                r={RADIUS}
                fill="none"
                stroke={seg.color}
                strokeWidth={STROKE}
                strokeDasharray={dashArray}
                strokeDashoffset={dashOffset}
                transform={`rotate(-90 ${CENTER} ${CENTER})`}
              />
            )
          })}
        </svg>
        <div className="donut-center">
          <strong>{centerLabel}</strong>
          {centerSub ? <span>{centerSub}</span> : null}
        </div>
      </div>
      <div className="kp-legend">
        {segments.map((seg) => (
          <div key={seg.key} className="legend-row">
            <span className="legend-dot" style={{ background: seg.color }} aria-hidden />
            <span className="legend-name">{seg.label}</span>
            <strong>{formatValue ? formatValue(seg.value) : String(seg.value)}</strong>
          </div>
        ))}
      </div>
    </div>
  )
}
