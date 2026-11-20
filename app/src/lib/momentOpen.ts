// Naming and opening a moment of the Overview: who it is with, and the day it
// opens on. Shared by the shelf of moments and the list of the quiet.
import { contactTitle } from './conversations'
import type { Moment } from './moments'
import type { Contact } from '../types'

// a moment as a place in time: one day of one conversation
export type MomentOfDay = {
  contact: Contact
  title: string
  // "YYYY-MM-DD"
  day: string
  // when the quoted message was sent, if the moment quotes one
  at: string | null
  // that message's id, where the server gives one
  quotedId?: string
  // what was clicked to open it, for the sheet to grow out of
  from?: Element
}

// a moment can come from a conversation outside the loaded list; it is named and opened the same way
function contactOf(moment: Moment, loaded: Map<string, Contact>): Contact {
  return loaded.get(moment.chatId) ?? { chat_id: moment.chatId, display_name: moment.savedName }
}

// "Sam in Family", the way a cited message names its source
export function momentTitle(moment: Moment, loaded: Map<string, Contact>) {
  const conversation = contactTitle(contactOf(moment, loaded))
  return moment.quotedFrom ? `${moment.quotedFrom} in ${conversation}` : conversation
}

// A moment is a place in time: it opens on its day. One from a build that
// did not say the day opens its conversation.
export function openMoment(
  moment: Moment,
  loaded: Map<string, Contact>,
  onOpenDay: (day: MomentOfDay) => void,
  onSelectContact: (contact: Contact) => void,
  from?: Element,
) {
  const contact = contactOf(moment, loaded)
  if (!moment.day) return onSelectContact(contact)
  onOpenDay({
    contact,
    title: contactTitle(contact),
    day: moment.day,
    at: moment.at ?? null,
    quotedId: moment.quotedId,
    from,
  })
}
