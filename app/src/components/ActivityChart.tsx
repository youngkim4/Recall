import { useState } from 'react'
import type { MonthlyPoint } from '../types'
import { formatNumber } from '../lib/format'

type ActivityChartProps = {
  data: MonthlyPoint[]
}

export function ActivityChart({ data }: ActivityChartProps) {
  const [activeIndex, setActiveIndex] = useState<number | null>(null)

  if (!data.length) {
    return <div className="chart-empty">No monthly data yet.</div>
  }

  const width = 760
  const height = 230
  const padding = { top: 18, right: 18, bottom: 48, left: 58 }
  const innerWidth = width - padding.left - padding.right
  const innerHeight = height - padding.top - padding.bottom
  const max = Math.max(1, ...data.map((point) => point.total))
  const step = innerWidth / Math.max(1, data.length - 1)
  const points = data.map((point, index) => {
    const x = padding.left + index * step
    const y = padding.top + innerHeight - (point.total / max) * innerHeight
    return { ...point, x, y }
  })
  const baseline = height - padding.bottom
  const linePath = smoothPath(points)
  const areaPath = points.length
    ? `${linePath} L ${points[points.length - 1].x.toFixed(2)},${baseline} L ${points[0].x.toFixed(2)},${baseline} Z`
    : ''
  const activePoint = activeIndex === null ? null : points[activeIndex]
  const tooltipWidth = 178
  const tooltipHeight = 78
  const tooltipX = activePoint ? Math.max(8, Math.min(width - tooltipWidth - 8, activePoint.x - tooltipWidth / 2)) : 0
  const tooltipY = activePoint
    ? activePoint.y - tooltipHeight - 14 < 8
      ? activePoint.y + 14
      : activePoint.y - tooltipHeight - 14
    : 0

  function updateActivePoint(clientX: number, svg: SVGSVGElement) {
    const rect = svg.getBoundingClientRect()
    const localX = ((clientX - rect.left) / rect.width) * width
    const nextIndex = Math.max(0, Math.min(points.length - 1, Math.round((localX - padding.left) / step)))
    setActiveIndex(nextIndex)
  }

  const yTicks = [0, Math.round(max / 2), max].filter(
    (tick, index, ticks) => ticks.indexOf(tick) === index,
  )
  const xTickEvery = Math.max(1, Math.ceil(data.length / 6))
  const xTicks = points.filter((_, index) => index % xTickEvery === 0 || index === points.length - 1)

  return (
    <figure className="activity-chart">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Monthly message activity"
        onPointerMove={(event) => updateActivePoint(event.clientX, event.currentTarget)}
        onPointerLeave={() => setActiveIndex(null)}
      >
        <defs>
          <linearGradient id="chartAreaFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#c13c27" stopOpacity="0.2" />
            <stop offset="60%" stopColor="#c13c27" stopOpacity="0.06" />
            <stop offset="100%" stopColor="#c13c27" stopOpacity="0" />
          </linearGradient>
        </defs>
        {yTicks.map((tick) => {
          const y = padding.top + innerHeight - (tick / max) * innerHeight
          return (
            <g key={tick}>
              <line className="chart-grid" x1={padding.left} x2={width - padding.right} y1={y} y2={y} />
              <text className="chart-tick-label" x={padding.left - 10} y={y + 4} textAnchor="end">
                {formatNumber(tick)}
              </text>
            </g>
          )
        })}
        <line
          className="chart-axis"
          x1={padding.left}
          x2={width - padding.right}
          y1={height - padding.bottom}
          y2={height - padding.bottom}
        />
        <line
          className="chart-axis"
          x1={padding.left}
          x2={padding.left}
          y1={padding.top}
          y2={height - padding.bottom}
        />
        <path className="chart-area" d={areaPath} />
        <path className="chart-line" d={linePath} fill="none" />
        {points.map((point, index) => (
          <g key={point.month}>
            <circle
              className={`chart-point ${activeIndex === index ? 'active' : ''}`}
              cx={point.x}
              cy={point.y}
              r="4"
            />
            <circle
              className="chart-hit-target"
              cx={point.x}
              cy={point.y}
              r="15"
              tabIndex={0}
              role="button"
              aria-label={`${point.month}: ${formatNumber(point.total)} messages, ${formatNumber(point.sent)} sent, ${formatNumber(point.received)} received`}
              onFocus={() => setActiveIndex(index)}
              onBlur={() => setActiveIndex(null)}
              onPointerEnter={() => setActiveIndex(index)}
            />
          </g>
        ))}
        {xTicks.map((point) => (
          <g key={`tick-${point.month}`}>
            <line
              className="chart-tick"
              x1={point.x}
              x2={point.x}
              y1={height - padding.bottom}
              y2={height - padding.bottom + 5}
            />
            <text className="chart-tick-label" x={point.x} y={height - padding.bottom + 20} textAnchor="middle">
              {point.month}
            </text>
          </g>
        ))}
        {activePoint ? (
          <g className="chart-tooltip" aria-hidden="true">
            <line
              className="chart-crosshair"
              x1={activePoint.x}
              x2={activePoint.x}
              y1={padding.top}
              y2={height - padding.bottom}
            />
            <circle className="chart-active-dot" cx={activePoint.x} cy={activePoint.y} r="5" />
            <rect x={tooltipX} y={tooltipY} width={tooltipWidth} height={tooltipHeight} rx="8" />
            <text className="chart-tooltip-title" x={tooltipX + 12} y={tooltipY + 22}>
              {activePoint.month}
            </text>
            <text x={tooltipX + 12} y={tooltipY + 43}>
              Total {formatNumber(activePoint.total)}
            </text>
            <text x={tooltipX + 12} y={tooltipY + 62}>
              Sent {formatNumber(activePoint.sent)} / Received {formatNumber(activePoint.received)}
            </text>
          </g>
        ) : null}
        <text className="chart-axis-title" x={(padding.left + width - padding.right) / 2} y={height - 8} textAnchor="middle">
          Month
        </text>
        <text
          className="chart-axis-title"
          x={-(padding.top + innerHeight / 2)}
          y={14}
          textAnchor="middle"
          transform="rotate(-90)"
        >
          Messages
        </text>
      </svg>
    </figure>
  )
}

// smooth (Catmull-Rom -> cubic bezier) path through the points, for a
// premium curved line instead of straight segments
function smoothPath(pts: Array<{ x: number; y: number }>) {
  if (pts.length < 2) return pts.length ? `M ${pts[0].x.toFixed(2)},${pts[0].y.toFixed(2)}` : ''
  const segments = pts.slice(1).map((p2, index) => {
    const p1 = pts[index]
    const p0 = pts[index - 1] ?? p1
    const p3 = pts[index + 2] ?? p2
    const cp1x = p1.x + (p2.x - p0.x) / 6
    const cp1y = p1.y + (p2.y - p0.y) / 6
    const cp2x = p2.x - (p3.x - p1.x) / 6
    const cp2y = p2.y - (p3.y - p1.y) / 6
    return `C ${cp1x.toFixed(2)},${cp1y.toFixed(2)} ${cp2x.toFixed(2)},${cp2y.toFixed(2)} ${p2.x.toFixed(2)},${p2.y.toFixed(2)}`
  })
  return `M ${pts[0].x.toFixed(2)},${pts[0].y.toFixed(2)} ${segments.join(' ')}`
}
