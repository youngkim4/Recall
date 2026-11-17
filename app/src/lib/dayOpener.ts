import { createContext, useContext } from 'react'
import type { MomentDay } from './momentDay'

// The way to open one day of a conversation in the day sheet, for whatever
// is drawn under a view that holds a sheet. Null where no sheet is near: a
// moment is then plain text.
export const DayOpener = createContext<((day: MomentDay) => void) | null>(null)

export function useDayOpener() {
  return useContext(DayOpener)
}
