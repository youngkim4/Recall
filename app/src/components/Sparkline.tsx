type SparklineProps = {
  data: number[]
  ariaLabel?: string
}

const WIDTH = 100
const HEIGHT = 34
const PAD = 2

// tiny inline area+line trend, for KPI cards where a real time series exists
export function Sparkline({ data, ariaLabel }: SparklineProps) {
  if (data.length < 2) return null

  const max = Math.max(...data)
  const min = Math.min(...data)
  const range = max - min || 1
  const step = WIDTH / (data.length - 1)

  const points = data.map((value, index) => {
    const x = index * step
    const y = HEIGHT - PAD - ((value - min) / range) * (HEIGHT - PAD * 2)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  })
  const line = points.join(' ')
  const area = `0,${HEIGHT} ${line} ${WIDTH},${HEIGHT}`

  return (
    <svg
      className="sparkline"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={ariaLabel}
    >
      <polygon className="sparkline-area" points={area} />
      <polyline className="sparkline-line" points={line} />
    </svg>
  )
}
