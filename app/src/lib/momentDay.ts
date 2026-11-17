// A moment of a report as a place in the archive: one day of one
// conversation, and the message that the moment quotes.
import type { AnalysisEvent } from '../types'
import { dayOfMessage } from './dayParts.ts'

export type MomentDay = {
  chatId: string
  // "YYYY-MM-DD"
  day: string
  // when the quoted message was sent; null to open the day at its start
  at: string | null
  // the id of that message, as a message of /api/day carries it
  pickedId?: string
}

const textOf = (value: unknown) => (typeof value === 'string' ? value : '')

// The day that an event of a report opens. Null for an event that does not
// say its conversation and its day: a report of an older build has neither,
// and its moments open nothing.
export function dayOfEvent(event: AnalysisEvent | null | undefined): MomentDay | null {
  const chatId = textOf(event?.chatId)
  const day = dayOfMessage(event?.day)
  if (!chatId || !day) return null
  return { chatId, day, at: textOf(event?.at) || null, pickedId: textOf(event?.pickedId) || undefined }
}

// readMoments in reportMoments.ts names each moment by the place of its event: "moment-<place>"
const PLACE = /^moment-(\d+)$/

// The day that a moment of the timeline opens, by the key readMoments gave it.
export function dayOfMoment(events: AnalysisEvent[], key: string): MomentDay | null {
  const place = PLACE.exec(key)
  return place ? dayOfEvent(events[Number(place[1])]) : null
}
