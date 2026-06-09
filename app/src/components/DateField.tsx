import { useEffect, useMemo, useRef, useState } from 'react'
import { CalendarIcon } from './Icons'

type DateFieldProps = {
  value: string // "YYYY-MM-DD" or ""
  onChange: (value: string) => void
  ariaLabel?: string
  placeholder?: string
}

const WEEKDAYS = ['Su', 'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa']
const MONTHS = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
]

function parseYMD(value: string): Date | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value || '')
  if (!match) return null
  const date = new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]))
  return Number.isNaN(date.getTime()) ? null : date
}

function toYMD(date: Date): string {
  const y = date.getFullYear()
  const m = String(date.getMonth() + 1).padStart(2, '0')
  const d = String(date.getDate()).padStart(2, '0')
  return `${y}-${m}-${d}`
}

function formatDisplay(value: string): string {
  const date = parseYMD(value)
  if (!date) return ''
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

function sameDay(a: Date, b: Date): boolean {
  return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate()
}

function monthGrid(view: Date): Date[] {
  const first = new Date(view.getFullYear(), view.getMonth(), 1)
  const start = new Date(first)
  start.setDate(first.getDate() - first.getDay())
  return Array.from({ length: 42 }, (_, index) => {
    const day = new Date(start)
    day.setDate(start.getDate() + index)
    return day
  })
}

export function DateField({ value, onChange, ariaLabel, placeholder = 'Select date' }: DateFieldProps) {
  const [open, setOpen] = useState(false)
  const [view, setView] = useState<Date>(() => parseYMD(value) ?? new Date())
  const rootRef = useRef<HTMLDivElement | null>(null)

  const selected = useMemo(() => parseYMD(value), [value])
  const today = useMemo(() => new Date(), [])
  const days = useMemo(() => monthGrid(view), [view])

  function toggleOpen() {
    if (open) {
      setOpen(false)
    } else {
      setView(parseYMD(value) ?? new Date())
      setOpen(true)
    }
  }

  useEffect(() => {
    if (!open) return
    function onPointer(event: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) setOpen(false)
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onPointer)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onPointer)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  function pick(day: Date) {
    onChange(toYMD(day))
    setOpen(false)
  }

  function shiftMonth(delta: number) {
    setView((current) => new Date(current.getFullYear(), current.getMonth() + delta, 1))
  }

  return (
    <div className="datefield" ref={rootRef}>
      <button
        type="button"
        className={`datefield-trigger ${value ? 'has-date' : ''} ${open ? 'is-open' : ''}`}
        aria-label={ariaLabel}
        aria-expanded={open}
        onClick={toggleOpen}
      >
        <span className={value ? 'datefield-value' : 'datefield-placeholder'}>
          {value ? formatDisplay(value) : placeholder}
        </span>
        <CalendarIcon className="calendar-icon" />
      </button>

      {open ? (
        <div className="calendar-pop" role="dialog" aria-label={ariaLabel}>
          <div className="calendar-head">
            <button type="button" className="calendar-nav" aria-label="Previous month" onClick={() => shiftMonth(-1)}>
              ‹
            </button>
            <strong>
              {MONTHS[view.getMonth()]} {view.getFullYear()}
            </strong>
            <button type="button" className="calendar-nav" aria-label="Next month" onClick={() => shiftMonth(1)}>
              ›
            </button>
          </div>
          <div className="calendar-weekdays">
            {WEEKDAYS.map((weekday) => (
              <span key={weekday}>{weekday}</span>
            ))}
          </div>
          <div className="calendar-grid">
            {days.map((day) => {
              const outside = day.getMonth() !== view.getMonth()
              const isSelected = selected ? sameDay(day, selected) : false
              const isToday = sameDay(day, today)
              const classes = [
                'calendar-day',
                outside ? 'outside' : '',
                isSelected ? 'selected' : '',
                isToday && !isSelected ? 'today' : '',
              ]
                .filter(Boolean)
                .join(' ')
              return (
                <button
                  key={toYMD(day)}
                  type="button"
                  className={classes}
                  aria-current={isToday ? 'date' : undefined}
                  aria-pressed={isSelected}
                  onClick={() => pick(day)}
                >
                  {day.getDate()}
                </button>
              )
            })}
          </div>
          {value ? (
            <div className="calendar-foot">
              <button type="button" className="calendar-clear" onClick={() => { onChange(''); setOpen(false) }}>
                Clear
              </button>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
