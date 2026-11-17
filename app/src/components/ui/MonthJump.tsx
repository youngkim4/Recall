import { useMemo, type CSSProperties, type KeyboardEvent } from 'react'
import { formatLongMonth, plural } from '../../lib/format'
import { byYear, nextMonth, type MonthCount } from '../../lib/months'

type MonthJumpProps = {
  // every month of the conversation, oldest first
  months: MonthCount[]
  // the month the calendar is on
  current: string
  // the months the chosen days reach, as "YYYY-MM", or null
  chosen: { from: string; to: string } | null
  onJump: (month: string) => void
}

const STEPS: Partial<Record<string, number>> = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -12, ArrowDown: 12 }
const initialFormat = new Intl.DateTimeFormat(undefined, { month: 'narrow' })
const INITIALS = Array.from({ length: 12 }, (_, index) => initialFormat.format(new Date(2024, index, 1)))

// The whole conversation at a glance: every month as a small bar, a row to a
// year. Pressing a month takes the calendar there.
export function MonthJump({ months, current, chosen, onJump }: MonthJumpProps) {
  const rows = useMemo(() => byYear(months), [months])
  const peak = useMemo(() => Math.max(1, ...months.map((entry) => entry.total)), [months])

  function handleKeyDown(event: KeyboardEvent<HTMLButtonElement>, month: string) {
    const step = STEPS[event.key]
    if (!step) return
    event.preventDefault()
    const target = nextMonth(months, month, step)
    event.currentTarget
      .closest('.jump')
      ?.querySelector<HTMLElement>(`[data-month="${target}"]`)
      ?.focus()
  }

  return (
    <div className="jump">
      <div className="jump-head" aria-hidden="true">
        <span />
        {INITIALS.map((letter, index) => (
          <span key={index}>{letter}</span>
        ))}
      </div>
      <div className="jump-years">
        {rows.map((row) => (
          <div key={row.year} className="jump-year">
            <span className="jump-year-name">{row.year}</span>
            {row.slots.map((slot, index) => {
              if (!slot) return <span key={index} />
              return (
                <button
                  key={slot.month}
                  type="button"
                  className="jump-month"
                  data-month={slot.month}
                  data-in={(chosen && slot.month >= chosen.from && slot.month <= chosen.to) || undefined}
                  aria-current={slot.month === current ? 'true' : undefined}
                  aria-label={`${formatLongMonth(slot.month)}, ${plural(slot.total, 'message')}`}
                  tabIndex={slot.month === current ? 0 : -1}
                  style={{ '--h': slot.total / peak } as CSSProperties}
                  onClick={() => onJump(slot.month)}
                  onKeyDown={(event) => handleKeyDown(event, slot.month)}
                >
                  <i />
                </button>
              )
            })}
          </div>
        ))}
      </div>
    </div>
  )
}
