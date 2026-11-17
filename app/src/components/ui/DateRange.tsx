import { useEffect, useId, useMemo, useRef, useState } from 'react'
import {
  countBetween,
  countOn,
  lastDay,
  monthTotals,
  orderDays,
  rangeLabel,
  tidyRange,
  type Daily,
  type DayRange,
} from '../../lib/days'
import { formatDate, formatDateRange, formatLongMonth, plural } from '../../lib/format'
import { WEEK_ORDER, WEEK_START, weekdayName } from '../../lib/locale'
import { shiftMonth } from '../../lib/months'
import { tabStopIn } from '../../lib/roving'
import { usePanel } from '../../lib/usePanel'
import { ChevronDownIcon, ChevronLeftIcon, ChevronRightIcon } from '../Icons'
import { CalendarMonth } from './CalendarMonth'
import { MonthJump } from './MonthJump'
import { Step } from './Step'

type DateRangeProps = {
  // "YYYY-MM-DD", or '' for no limit on that side
  since: string
  until: string
  // how many messages each day holds; no days while that is not known
  daily: Daily
  onChange: (range: DayRange) => void
}

const WEEKDAYS = WEEK_ORDER.map((weekday) => weekdayName(weekday, 'narrow'))
const ALL: DayRange = { since: '', until: '' }

// A button that says which dates are covered, and a calendar to change them.
// Press a day and then another, or drag from one to the other; the range
// shows as it is made and nothing changes until it is complete. The month's
// name opens the whole conversation by month, to move a long way at once.
// Reset goes back to all of the dates.
export function DateRange({ since, until, daily, onChange }: DateRangeProps) {
  const id = useId()
  const known = daily.counts.length > 0
  const first = daily.start
  const last = lastDay(daily)
  // the day a drag started on
  const pressed = useRef<string | null>(null)
  // a day the keyboard moved to in a month not yet drawn, focused once it is
  const toFocus = useRef<string | null>(null)
  const title = useRef<HTMLButtonElement>(null)
  // the month on show; until one is chosen it is the conversation's last
  const [viewed, setView] = useState('')
  const view = viewed || last.slice(0, 7)
  const [slide, setSlide] = useState(0)
  const [jumping, setJumping] = useState(false)
  // the first day of a choice that is waiting for its second
  const [anchor, setAnchor] = useState<string | null>(null)
  // the day under the pointer, or under keyboard focus
  const [hover, setHover] = useState<string | null>(null)
  const { control, panel, open, close, panelEvents } = usePanel<HTMLDivElement>({
    onToggle(opening) {
      // a choice left half made is dropped, and the calendar opens on the chosen days
      setAnchor(null)
      setHover(null)
      setJumping(false)
      setSlide(0)
      pressed.current = null
      toFocus.current = null
      // dates kept from another conversation can lie outside this one
      const wanted = (until || last).slice(0, 7)
      const firstMonth = first.slice(0, 7)
      const lastMonth = last.slice(0, 7)
      setView(opening ? (wanted < firstMonth ? firstMonth : wanted > lastMonth ? lastMonth : wanted) : '')
    },
  })

  // the day's button in the month on show; a month on its way out is not looked in
  const dayButton = (day: string) =>
    panel.current?.querySelector<HTMLElement>(`.step > :not([data-step='out']) [data-day="${day}"]`) ?? null

  useEffect(() => {
    if (!toFocus.current) return
    dayButton(toFocus.current)?.focus()
    toFocus.current = null
  })

  const months = useMemo(() => monthTotals(daily), [daily])
  const range = { since, until }
  const chosen = since || until ? { since: since || first, until: until || last } : null
  const draft = anchor ? orderDays(anchor, hover ?? anchor) : null
  const shown = draft ?? chosen
  // dates kept from another conversation can end outside this one, where no day takes the focus
  const tabStop = tabStopIn(view, [hover, anchor, chosen?.until], first, last)
  const readout = draft
    ? {
        name: draft.since === draft.until ? formatDate(draft.since) : formatDateRange(draft.since, draft.until),
        total: countBetween(daily, draft.since, draft.until),
      }
    : hover
      ? { name: formatDate(hover), total: countOn(daily, hover) }
      : {
          // the days themselves; the button already carries the short name
          name: formatDateRange(chosen?.since ?? first, chosen?.until ?? last),
          total: countBetween(daily, since, until),
        }

  function commit(next: DayRange) {
    const tidy = tidyRange(next, daily)
    if (tidy.since !== since || tidy.until !== until) onChange(tidy)
    close()
  }

  // a range with nothing in it cannot be read, so it is not taken
  function take(next: DayRange) {
    if (countBetween(daily, next.since, next.until) > 0) commit(next)
  }

  function pick(day: string) {
    if (anchor === null) setAnchor(day)
    else take(orderDays(anchor, day))
  }

  function release(day: string) {
    const from = pressed.current
    pressed.current = null
    // let go on another day, the press was a drag; the click that follows lands on neither
    if (from && from !== day) take(orderDays(from, day))
  }

  function handleHover(day: string | null, dragging: boolean) {
    setHover(day)
    if (day && dragging && pressed.current && pressed.current !== day) setAnchor(pressed.current)
  }

  function step(by: number) {
    setSlide(by)
    setView(shiftMonth(view, by))
  }

  // the keyboard moved to a day: bring its month into view with no motion
  function moveTo(day: string) {
    const held = day < first ? first : day > last ? last : day
    const drawn = dayButton(held)
    // a day already on show takes the focus now; one in another month, once that month is drawn
    if (drawn) drawn.focus()
    else toFocus.current = held
    setSlide(0)
    setView(held.slice(0, 7))
    setHover(held)
  }

  const canGoEarlier = !jumping && view > first.slice(0, 7)
  const canGoLater = !jumping && view < last.slice(0, 7)

  return (
    <>
      <button
        ref={control}
        type="button"
        className="picker"
        popoverTarget={id}
        disabled={!known}
        aria-haspopup="dialog"
        aria-expanded={open}
      >
        {/* a new choice fades in over the old one */}
        <span key={`${since} ${until}`} className="picker-value swap">
          {rangeLabel(range, daily)}
        </span>
        <ChevronDownIcon />
      </button>
      <div ref={panel} id={id} popover="auto" className="popover cal" role="dialog" aria-label="Dates" {...panelEvents}>
        {known ? (
          <>
            <div className="cal-bar">
              <button
                ref={title}
                type="button"
                className="cal-title"
                aria-expanded={jumping}
                aria-label={`${formatLongMonth(view)}, show every month`}
                onClick={() => setJumping((value) => !value)}
              >
                {formatLongMonth(view)}
                <ChevronDownIcon />
              </button>
              <button
                type="button"
                className="icon-btn sm"
                aria-label="Earlier month"
                aria-disabled={!canGoEarlier}
                onClick={() => {
                  if (canGoEarlier) step(-1)
                }}
              >
                <ChevronLeftIcon />
              </button>
              <button
                type="button"
                className="icon-btn sm"
                aria-label="Later month"
                aria-disabled={!canGoLater}
                onClick={() => {
                  if (canGoLater) step(1)
                }}
              >
                <ChevronRightIcon />
              </button>
            </div>
            <div className="cal-body">
              {jumping ? (
                <MonthJump
                  months={months}
                  current={view}
                  chosen={chosen ? { from: chosen.since.slice(0, 7), to: chosen.until.slice(0, 7) } : null}
                  onJump={(month) => {
                    setSlide(0)
                    setView(month)
                    setJumping(false)
                    // the month that was pressed goes away, and the focus would go with it
                    title.current?.focus()
                  }}
                />
              ) : (
                <>
                  <div className="cal-week" aria-hidden="true">
                    {WEEKDAYS.map((letter, index) => (
                      <span key={index}>{letter}</span>
                    ))}
                  </div>
                  <Step
                    id={view}
                    slide={slide}
                    render={(month) => (
                      <CalendarMonth
                        month={month}
                        weekStart={WEEK_START}
                        daily={daily}
                        first={first}
                        last={last}
                        shown={shown}
                        tabStop={month === view ? tabStop : null}
                        onPick={pick}
                        onPress={(day) => {
                          pressed.current = day
                        }}
                        onRelease={release}
                        onHover={handleHover}
                        onMove={moveTo}
                      />
                    )}
                  />
                </>
              )}
            </div>
            <div className="cal-foot">
              <p className="cal-readout" aria-hidden="true">
                <span>{readout.name}</span>
                <span>{readout.total ? plural(readout.total, 'message') : 'No messages'}</span>
              </p>
              {/* only when some dates are left out: there is nothing to reset otherwise */}
              {chosen ? (
                <button type="button" className="btn" onClick={() => commit(ALL)}>
                  Reset
                </button>
              ) : null}
            </div>
          </>
        ) : null}
      </div>
    </>
  )
}
