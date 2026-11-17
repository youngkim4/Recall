import { useMemo, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { formatNumber, plural } from '../../lib/format'
import { busiestHour, heatLevel, heatSteps } from '../../lib/heat'
import { hourName, WEEK_ORDER, weekdayName } from '../../lib/locale'
import { HeatLegend } from './HeatLegend'

const HOURS = 24
// the hours named under the grid
const NAMED_HOURS = [0, 6, 12, 18]
const STEPS: Partial<Record<string, [number, number]>> = {
  ArrowLeft: [0, -1],
  ArrowRight: [0, 1],
  ArrowUp: [-1, 0],
  ArrowDown: [1, 0],
}

// the counts keep Monday in row 0; Date numbers Sunday 0
const rowOf = (weekday: number) => (weekday + 6) % 7

type Cell = { row: number; hour: number }

// The hours of the week: a row a weekday, a column an hour, each darker the
// more messages it holds over the whole archive. Moving across it, by pointer
// or arrow keys, reads an hour.
export function Rhythm({ rhythm }: { rhythm: number[][] }) {
  // row is a place in the reader's week, so row 0 is the day their week starts on
  const [active, setActive] = useState<Cell | null>(null)
  // the hour the keyboard last read; the pointer reads an hour without saying it aloud
  const [said, setSaid] = useState<Cell | null>(null)
  const steps = useMemo(() => heatSteps(rhythm.flat()), [rhythm])
  const busiest = useMemo(() => busiestHour(rhythm), [rhythm])
  const countAt = (cell: Cell) => rhythm[rowOf(WEEK_ORDER[cell.row])]?.[cell.hour] ?? 0
  const nameOf = (weekday: number, hour: number) => `${weekdayName(weekday, 'long')}, ${hourName(hour)}`

  function cellAt(event: PointerEvent<HTMLDivElement>): Cell {
    const box = event.currentTarget.getBoundingClientRect()
    return {
      row: Math.max(0, Math.min(6, Math.floor(((event.clientY - box.top) / box.height) * 7))),
      hour: Math.max(0, Math.min(HOURS - 1, Math.floor(((event.clientX - box.left) / box.width) * HOURS))),
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === 'Escape') {
      setActive(null)
      return
    }
    const step = STEPS[event.key]
    if (!step) return
    event.preventDefault()
    // the first press of an arrow lands on the busiest hour
    const from = active ?? {
      row: busiest ? WEEK_ORDER.indexOf((busiest.weekday + 1) % 7) : 0,
      hour: busiest?.hour ?? 0,
    }
    const cell =
      active === null
        ? from
        : {
            row: Math.max(0, Math.min(6, from.row + step[0])),
            hour: Math.max(0, Math.min(HOURS - 1, from.hour + step[1])),
          }
    setActive(cell)
    setSaid(cell)
  }

  if (!busiest) return null

  return (
    <section className="section" aria-labelledby="hours-title">
      <div className="section-head">
        <h2 id="hours-title">By hour</h2>
        <p className="heat-readout">
          {active ? (
            <>
              <span>{nameOf(WEEK_ORDER[active.row], active.hour)}</span>
              <b>{formatNumber(countAt(active))}</b>
            </>
          ) : (
            <>
              <span>Busiest {nameOf((busiest.weekday + 1) % 7, busiest.hour)}</span>
              <b>{formatNumber(busiest.count)}</b>
            </>
          )}
        </p>
      </div>
      <div className="card card-pad">
        <div className="hours">
          <div className="hours-weekdays" aria-hidden="true">
            {WEEK_ORDER.map((weekday) => (
              <span key={weekday}>{weekdayName(weekday, 'short')}</span>
            ))}
          </div>
          <div
            className="hours-grid"
            tabIndex={0}
            role="group"
            aria-label="Messages in each hour of the week. Use the arrow keys to read an hour."
            onPointerMove={(event) => {
              const cell = cellAt(event)
              setActive((current) => (current?.row === cell.row && current.hour === cell.hour ? current : cell))
            }}
            onPointerLeave={() => setActive(null)}
            onKeyDown={handleKeyDown}
            onBlur={() => setActive(null)}
          >
            {WEEK_ORDER.flatMap((weekday, row) =>
              Array.from({ length: HOURS }, (_, hour) => (
                <i
                  key={`${weekday}-${hour}`}
                  data-level={heatLevel(rhythm[rowOf(weekday)]?.[hour] ?? 0, steps)}
                  data-active={(active?.row === row && active.hour === hour) || undefined}
                />
              )),
            )}
          </div>
          <div className="hours-names" aria-hidden="true">
            {NAMED_HOURS.map((hour) => (
              <span key={hour} style={{ gridColumn: hour + 1 }}>
                {hourName(hour)}
              </span>
            ))}
          </div>
        </div>
        <div className="heat-foot">
          <HeatLegend />
        </div>
      </div>
      <p className="sr-only" aria-live="polite">
        {said ? `${nameOf(WEEK_ORDER[said.row], said.hour)}, ${plural(countAt(said), 'message')}` : ''}
      </p>
    </section>
  )
}
