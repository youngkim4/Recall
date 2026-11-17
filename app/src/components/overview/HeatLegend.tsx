// What the five shades of a heat map stand for.
export function HeatLegend() {
  return (
    <span className="heat-legend" aria-hidden="true">
      Less
      {[0, 1, 2, 3, 4].map((level) => (
        <i key={level} data-level={level} />
      ))}
      More
    </span>
  )
}
