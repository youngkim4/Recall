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
  size?: number
}

// SVG ring with a center total + legend (Tremor DonutChart)
export function Donut({ segments, centerLabel, centerSub, formatValue, size = 132 }: DonutProps) {
  const radius = size * 0.41
  const stroke = size * 0.106
  const center = size / 2
  const circ = 2 * Math.PI * radius

  const sum = segments.reduce((total, seg) => total + seg.value, 0) || 1
  const lengths = segments.map((seg) => (seg.value / sum) * circ)
  // cumulative start offset per segment, computed immutably
  const offsets = lengths.map((_, index) => lengths.slice(0, index).reduce((a, b) => a + b, 0))

  return (
    <div className="donut">
      <div className="donut-svg" style={{ width: size, height: size }}>
        <svg viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Share by segment">
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            stroke="var(--surface-3)"
            strokeWidth={stroke}
          />
          {segments.map((seg, index) => {
            const length = lengths[index]
            const dashArray = `${length} ${circ - length}`
            const dashOffset = -offsets[index]
            return (
              <circle
                key={seg.key}
                cx={center}
                cy={center}
                r={radius}
                fill="none"
                stroke={seg.color}
                strokeWidth={stroke}
                strokeDasharray={dashArray}
                strokeDashoffset={dashOffset}
                transform={`rotate(-90 ${center} ${center})`}
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
