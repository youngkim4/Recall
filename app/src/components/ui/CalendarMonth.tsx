import { useMemo, type CSSProperties, type KeyboardEvent } from 'react'
import { addDays, addMonths, countOn, weeksOf, type Daily, type DayRange } from '../../lib/days'
import { plural } from '../../lib/format'

type CalendarMonthProps = {
  // "YYYY-MM"
  month: string
  // 0 is Sunday
  weekStart: number
  daily: Daily
  // the first and last day that can be chosen
  first: string
  last: string
  // the days drawn as chosen: a made choice, or one being made
  shown: DayRange | null
  // the one day Tab stops on
  tabStop: string | null
  onPick: (day: string) => void
  onPress: (day: string) => void
  onRelease: (day: string) => void
  onHover: (day: string | null, dragging: boolean) => void
  // the keyboard asks for another day, which may be in another month
  onMove: (day: string) => void
}

const STEPS: Partial<Record<string, number>> = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -7, ArrowDown: 7 }
const labelFormat = new Intl.DateTimeFormat(undefined, { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })
const localDate = (day: string) => new Date(Number(day.slice(0, 4)), Number(day.slice(5, 7)) - 1, Number(day.slice(8, 10)))

// One month of days. A day with messages carries a mark as wide as the day
// was busy; a day outside the conversation is not offered. The chosen days
// share one band, with a solid cap on the first and the last.
export function CalendarMonth({
  month,
  weekStart,
  daily,
  first,
  last,
  shown,
  tabStop,
  onPick,
  onPress,
  onRelease,
  onHover,
  onMove,
}: CalendarMonthProps) {
  const weeks = useMemo(() => weeksOf(month, weekStart), [month, weekStart])
  // marks are sized against the busiest day of the month in view
  const peak = useMemo(
    () => Math.max(1, ...weeks.flat().map((day) => (day ? countOn(daily, day) : 0))),
    [weeks, daily],
  )

  function handleKeyDown(event: KeyboardEvent<HTMLButtonElement>, day: string) {
    const step = STEPS[event.key]
    const column = weeks.find((week) => week.includes(day))?.indexOf(day) ?? 0
    const target = step
      ? addDays(day, step)
      : event.key === 'PageUp'
        ? addMonths(day, -1)
        : event.key === 'PageDown'
          ? addMonths(day, 1)
          : event.key === 'Home'
            ? addDays(day, -column)
            : event.key === 'End'
              ? addDays(day, 6 - column)
              : null
    if (!target) return
    event.preventDefault()
    onMove(target)
  }

  return (
    <div className="cal-days" onPointerLeave={() => onHover(null, false)}>
      {weeks.flat().map((day, index) => {
        if (!day) return <span key={index} />
        const number = Number(day.slice(8, 10))
        if (day < first || day > last) {
          return (
            <span key={day} className="cal-day is-out" aria-hidden="true">
              <span className="cal-num">{number}</span>
            </span>
          )
        }
        const count = countOn(daily, day)
        const inShown = Boolean(shown && day >= shown.since && day <= shown.until)
        const starts = inShown && day === shown?.since
        const ends = inShown && day === shown?.until
        return (
          <button
            key={day}
            type="button"
            className="cal-day"
            data-day={day}
            data-in={inShown || undefined}
            data-edge={starts && ends ? 'both' : starts ? 'start' : ends ? 'end' : undefined}
            data-quiet={count === 0 || undefined}
            aria-label={`${labelFormat.format(localDate(day))}, ${count ? plural(count, 'message') : 'no messages'}`}
            aria-pressed={inShown}
            tabIndex={day === tabStop ? 0 : -1}
            // the square root keeps a quiet day's mark from vanishing beside a busy one
            style={{ '--v': Math.sqrt(count / peak) } as CSSProperties}
            onPointerDown={(event) => {
              if (event.button === 0) onPress(day)
            }}
            onPointerEnter={(event) => onHover(day, event.buttons === 1)}
            onPointerUp={() => onRelease(day)}
            onClick={() => onPick(day)}
            // the readout follows the keyboard only while the keyboard is in use
            onFocus={(event) => {
              if (event.currentTarget.matches(':focus-visible')) onHover(day, false)
            }}
            onKeyDown={(event) => handleKeyDown(event, day)}
          >
            <span className="cal-num">{number}</span>
            {count ? <i /> : null}
          </button>
        )
      })}
    </div>
  )
}
