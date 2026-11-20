import { createContext, useContext } from 'react'
import type { MomentDay } from './momentDay'

// The way to open one day of a conversation in the day sheet, for whatever
// is drawn under a view that holds a sheet. Null where no sheet is near: a
// moment is then plain text. `from` is what was clicked, for the sheet to grow out of.
export type OpenDay = (day: MomentDay, from?: Element) => void
export const DayOpener = createContext<OpenDay | null>(null)

export function useDayOpener() {
  return useContext(DayOpener)
}
