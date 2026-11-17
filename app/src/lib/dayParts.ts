// A long day is read in parts: the part around a message first, then the part
// before what is shown or the part after it, each time a button is pressed.

import type { DayMessages, SearchResult } from '../types'

// how many more messages of a long day one press brings
export const PART = 200

// The day a message was sent, "YYYY-MM-DD", from its time as the server
// writes it. Null for a message with no time, or with one that is not a
// date: such a message is in no day, and has no day to open.
export function dayOfMessage(timestamp: string | null | undefined) {
  return typeof timestamp === 'string' && /^\d{4}-\d{2}-\d{2}/.test(timestamp) ? timestamp.slice(0, 10) : null
}

export type PartAsk = { offset: number; limit: number }

type Shown = Pick<DayMessages, 'offset' | 'total' | 'messages'>

// The part of a day before what is shown, or the part after it: where it
// starts and how many messages it holds. Null when that side has no more.
// It asks for no more than is there, so the place asked for is the place given.
export function partToAsk(shown: Shown, earlier: boolean, size = PART): PartAsk | null {
  if (earlier) {
    const from = Math.max(0, shown.offset - size)
    return from < shown.offset ? { offset: from, limit: shown.offset - from } : null
  }
  const end = shown.offset + shown.messages.length
  const limit = Math.min(size, shown.total - end)
  return limit > 0 ? { offset: end, limit } : null
}

// What is shown with a part joined to it. A part that does not meet what is
// shown is left out: it came twice, or what is shown changed on its way here.
export function joinPart(shown: DayMessages, part: DayMessages, earlier: boolean): DayMessages {
  if (!part.messages.length) return shown
  if (earlier) {
    if (part.offset + part.messages.length !== shown.offset) return shown
    return { ...shown, offset: part.offset, messages: [...part.messages, ...shown.messages] }
  }
  if (part.offset !== shown.offset + shown.messages.length) return shown
  return { ...shown, messages: [...shown.messages, ...part.messages] }
}

// Which message of a day a moment quotes: the one with its id, or failing
// that the first one sent at its time. -1 when the day does not hold it.
export function quotedIndex(
  messages: Array<Pick<SearchResult, 'messageId' | 'timestamp'>>,
  id: string | null | undefined,
  at: string | null | undefined,
) {
  const byId = id ? messages.findIndex((message) => message.messageId === id) : -1
  if (byId >= 0) return byId
  return at ? messages.findIndex((message) => message.timestamp === at) : -1
}
