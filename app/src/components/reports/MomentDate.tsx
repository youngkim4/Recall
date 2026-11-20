import type { ReactNode } from 'react'
import { useDayOpener } from '../../lib/dayOpener'
import type { MomentDay } from '../../lib/momentDay'

type MomentDateProps = {
  // the day of the archive that the moment is of; null for a moment that opens nothing
  opens: MomentDay | null
  // the date as it is written
  children: ReactNode
}

// The date of a moment in a report. Where the report says which day of which
// conversation the moment is of, the date is the way into that day. A moment
// of an older report says neither, and its date is only a date.
export function MomentDate({ opens, children }: MomentDateProps) {
  const openDay = useDayOpener()
  if (!opens || !openDay) return children
  return (
    <button type="button" className="link in-text" onClick={(event) => openDay(opens, event.currentTarget)}>
      <span className="sr-only">Open the day of </span>
      {children}
    </button>
  )
}
