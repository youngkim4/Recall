import { useState, type KeyboardEvent } from 'react'
import { plural } from '../../lib/format'
import { busiestHour, weekGrid, type WeekRow } from '../../lib/habits'
import type { StatsRow } from '../../lib/insights'
import { hourName, WEEK_ORDER, weekdayName } from '../../lib/locale'

type WhenGridProps = {
  // the answer by hour of the week (`weekday_hour`), or null while it loads
  rows: StatsRow[] | null
}

const HOURS = 24
// the hours named over the grid
const EVERY_HOURS = 3
// a cell's shade: none for an empty hour, then seven steps to the busiest
const STEPS = 7

const stepOf = (total: number, level: number) => (total ? 1 + Math.min(STEPS - 1, Math.floor(level * STEPS)) : 0)
// the server counts weekdays from Monday; Date from Sunday
const dateWeekday = (serverWeekday: number) => (serverWeekday + 1) % 7

type Spot = { row: number; hour: number }

function readout(grid: WeekRow[], spot: Spot | null, rows: StatsRow[]) {
  if (spot) {
    const day = grid[spot.row]
    const cell = day.cells[spot.hour]
    return `${weekdayName(day.weekday, 'long')} at ${hourName(spot.hour)}: ${plural(cell.total, 'message')}`
  }
  const top = busiestHour(rows)
  return top
    ? `Busiest: ${weekdayName(dateWeekday(top.weekday), 'long')} at ${hourName(top.hour)}, ${plural(top.total, 'message')}`
    : ''
}

// When the messages of the answer are sent, after GitHub's grid of
// contributions and Toggl's week: a row a day in the reader's order, a cell
// an hour, its shade one hue of ink from empty to the busiest hour. Moving
// over it, by pointer or arrow keys, reads one hour in the line over the
// grid, where the busiest hour stands until then.
export function WhenGrid({ rows }: WhenGridProps) {
  const [spot, setSpot] = useState<Spot | null>(null)
  const grid = weekGrid(rows ?? [], WEEK_ORDER)

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const moves: Record<string, [number, number]> = {
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
    }
    if (event.key === 'Escape') return setSpot(null)
    const move = moves[event.key]
    if (!move) return
    event.preventDefault()
    const from = spot ?? { row: 0, hour: 0 }
    setSpot({
      row: Math.max(0, Math.min(grid.length - 1, from.row + (spot ? move[0] : 0))),
      hour: Math.max(0, Math.min(HOURS - 1, from.hour + (spot ? move[1] : 0))),
    })
  }

  return (
    <section className="section insights-when" aria-labelledby="insights-when-title">
      <div className="section-head">
        <h2 id="insights-when-title">When</h2>
        <span className="meta when-readout" aria-live="polite">
          {rows ? readout(grid, spot, rows) : <span className="skeleton">Busiest: Wednesday at 5 PM, 0,000</span>}
        </span>
      </div>
      <div
        className="when-grid"
        role="img"
        aria-label="Messages by day of the week and hour. Use the arrow keys to read each hour."
        tabIndex={0}
        // a click reads an hour; it does not take the focus, so no ring is drawn
        onMouseDown={(event) => event.preventDefault()}
        onKeyDown={handleKeyDown}
        onBlur={() => setSpot(null)}
        onPointerLeave={() => setSpot(null)}
        data-reading={spot ? '' : undefined}
      >
        <span aria-hidden="true" />
        {Array.from({ length: HOURS }, (_, hour) => (
          <span key={hour} className="when-hour" aria-hidden="true" data-major={hour % 6 === 0 || undefined}>
            {hour % EVERY_HOURS === 0 ? hourName(hour) : ''}
          </span>
        ))}
        {grid.map((day, row) => [
          <span key={`day-${day.weekday}`} className="when-day" aria-hidden="true">
            {weekdayName(day.weekday, 'short')}
          </span>,
          ...day.cells.map((cell) => (
            <span
              key={`${day.weekday}-${cell.hour}`}
              className="when-cell"
              aria-hidden="true"
              data-step={stepOf(cell.total, cell.level)}
              data-active={spot?.row === row && spot.hour === cell.hour ? '' : undefined}
              onPointerEnter={() => setSpot({ row, hour: cell.hour })}
            />
          )),
        ])}
      </div>
    </section>
  )
}
