import { useMemo, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { pointName } from '../../lib/activity'
import { addDays, countOn, lastDay, type Daily } from '../../lib/days'
import { formatDate, formatNumber, plural } from '../../lib/format'
import { heatLevel, heatSteps, monthMarks, weeksEnding, type HeatSteps } from '../../lib/heat'
import { WEEK_ORDER, WEEK_START, weekdayName } from '../../lib/locale'
import { ChevronLeftIcon, ChevronRightIcon } from '../Icons'
import { Step } from '../ui/Step'
import { HeatLegend } from './HeatLegend'

const WEEKS = 53
// a month's name is wider than a week; one that starts this near the end is set against the edge
const LAST_WEEKS = 2
// stepping back a whole number of weeks keeps each weekday on its row
const YEAR_OF_DAYS = 52 * 7
const STEPS: Partial<Record<string, number>> = { ArrowLeft: -7, ArrowRight: 7, ArrowUp: -1, ArrowDown: 1 }

// the last day in view, that many years back from the newest
const endOf = (daily: Daily, back: number) => addDays(lastDay(daily), -back * YEAR_OF_DAYS)

// the first of the busiest days among these
function busiestOf(daily: Daily, days: Array<string | null>) {
  return days.reduce<{ day: string; count: number } | null>((best, day) => {
    const count = day === null ? 0 : countOn(daily, day)
    return day !== null && count > (best?.count ?? 0) ? { day, count } : best
  }, null)
}

// A year of days, a column a week: each day is darker the more messages it
// holds, on one scale for the whole archive so that years compare. Moving
// across it, by pointer or arrow keys, reads a day. The arrows step a year,
// and the year in view arrives from the side it lies on.
export function YearOfDays({ daily }: { daily: Daily }) {
  // how many years back from the newest the window sits, and the side it came from
  const [view, setView] = useState({ back: 0, slide: 0 })
  const [active, setActive] = useState<string | null>(null)
  // the day the keyboard last read; the pointer reads a day without saying it aloud
  const [said, setSaid] = useState<string | null>(null)
  const first = daily.start
  const end = endOf(daily, view.back)
  const weeks = useMemo(() => weeksEnding(end, WEEKS, WEEK_START), [end])
  const steps = useMemo(() => heatSteps(daily.counts), [daily])
  const busiest = useMemo(() => busiestOf(daily, weeks.flat()), [daily, weeks])
  const start = weeks[0][0] ?? first
  const since = start < first ? first : start
  const period = since.slice(0, 4) === end.slice(0, 4) ? end.slice(0, 4) : `${since.slice(0, 4)}–${end.slice(2, 4)}`
  const canGoEarlier = addDays(end, -YEAR_OF_DAYS) >= first
  const canGoLater = view.back > 0

  // an earlier year lies to the left, so it arrives from there
  function step(years: number) {
    setView((current) => ({ back: current.back + years, slide: -years }))
    setActive(null)
    setSaid(null)
  }

  return (
    <section className="section ruled" aria-labelledby="days-title">
      <div className="section-head">
        <div className="section-title">
          <h2 id="days-title">By day</h2>
          {/* the arrows sit on either side of the years they change, and keep their places as the years do */}
          <span className="stepper">
            <button
              type="button"
              className="icon-btn sm"
              aria-label="A year earlier"
              aria-disabled={!canGoEarlier}
              onClick={() => {
                if (canGoEarlier) step(1)
              }}
            >
              <ChevronLeftIcon />
            </button>
            <span key={period} className="stepper-value swap">
              {period}
            </span>
            <button
              type="button"
              className="icon-btn sm"
              aria-label="A year later"
              aria-disabled={!canGoLater}
              onClick={() => {
                if (canGoLater) step(-1)
              }}
            >
              <ChevronRightIcon />
            </button>
          </span>
        </div>
        <p className="heat-readout">
          {active ? (
            <>
              <span>{pointName(active, 'day')}</span>
              <b>{formatNumber(countOn(daily, active))}</b>
            </>
          ) : busiest ? (
            <>
              <span>Busiest {formatDate(busiest.day)}</span>
              <b>{formatNumber(busiest.count)}</b>
            </>
          ) : null}
        </p>
      </div>
      <div className="heat">
        <div className="days">
          <div className="days-weekdays" aria-hidden="true">
            {WEEK_ORDER.map((weekday, row) => (
              <span key={weekday}>{row % 2 === 1 ? weekdayName(weekday, 'short') : ''}</span>
            ))}
          </div>
          {/* in a narrow window the newest weeks stay in view and the oldest slide out of it */}
          <div className="days-view">
            <Step
              id={view.back}
              slide={view.slide}
              className="days-step"
              render={(back) => (
                <YearGrid
                  daily={daily}
                  end={endOf(daily, back)}
                  steps={steps}
                  active={back === view.back ? active : null}
                  onActive={setActive}
                  onSay={setSaid}
                />
              )}
            />
          </div>
        </div>
        <div className="heat-foot">
          <HeatLegend />
        </div>
      </div>
      <p className="sr-only" aria-live="polite">
        {said
          ? `${pointName(said, 'day')}, ${plural(countOn(daily, said), 'message')}`
          : `${formatDate(since)} to ${formatDate(end)}`}
      </p>
    </section>
  )
}

type GridProps = {
  daily: Daily
  end: string
  steps: HeatSteps
  active: string | null
  onActive: (day: string | null) => void
  // the day to read aloud, when the keyboard moved to it
  onSay: (day: string | null) => void
}

// The 53 weeks that end on a day: the names of the months over the cells.
function YearGrid({ daily, end, steps, active, onActive, onSay }: GridProps) {
  const first = daily.start
  const last = lastDay(daily)
  const weeks = useMemo(() => weeksEnding(end, WEEKS, WEEK_START), [end])
  const marks = useMemo(() => monthMarks(weeks), [weeks])
  const start = weeks[0][0] ?? first
  const since = start < first ? first : start
  const inArchive = (day: string | null): day is string => day !== null && day >= first && day <= last

  function dayAt(event: PointerEvent<HTMLDivElement>) {
    const box = event.currentTarget.getBoundingClientRect()
    const week = Math.floor(((event.clientX - box.left) / box.width) * WEEKS)
    const place = Math.floor(((event.clientY - box.top) / box.height) * 7)
    const day = weeks[Math.max(0, Math.min(WEEKS - 1, week))]?.[Math.max(0, Math.min(6, place))] ?? null
    return inArchive(day) ? day : null
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === 'Escape') {
      onActive(null)
      return
    }
    const step = STEPS[event.key]
    if (!step) return
    event.preventDefault()
    // the first press of an arrow lands on the newest day in view
    const wanted = active === null ? end : addDays(active, step)
    const day = wanted < since ? since : wanted > end ? end : wanted
    onActive(day)
    onSay(day)
  }

  return (
    <div className="days-year">
      <div className="days-months" aria-hidden="true">
        {marks.map((mark) => (
          <span
            key={mark.week}
            className={mark.week >= WEEKS - LAST_WEEKS ? 'is-last' : undefined}
            style={{ gridColumn: mark.week + 1 }}
          >
            {mark.text}
          </span>
        ))}
      </div>
      <div
        className="days-grid"
        tabIndex={0}
        role="group"
        aria-label={`Messages each day from ${formatDate(since)} to ${formatDate(end)}. Use the arrow keys to read a day.`}
        onPointerMove={(event) => onActive(dayAt(event))}
        onPointerLeave={() => onActive(null)}
        onKeyDown={handleKeyDown}
        onBlur={() => onActive(null)}
      >
        {weeks.flat().map((day, index) =>
          inArchive(day) ? (
            <i key={day} data-level={heatLevel(countOn(daily, day), steps)} data-active={day === active || undefined} />
          ) : (
            <span key={`none-${index}`} />
          ),
        )}
      </div>
    </div>
  )
}
