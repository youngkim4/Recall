import type { Contact } from '../types'
import { contactTitle, messageCount } from './conversations'
import { formatNumber } from './format'

// one choice in a list that is picked from
export type PickerOption = {
  value: string
  label: string
  // a quiet figure at the end of the row, such as a count
  meta?: string
}

// The choices for narrowing a question or a search to one conversation:
// everything first, then each conversation with how many messages it holds.
export function scopeChoices(contacts: Contact[]): PickerOption[] {
  return [
    { value: '', label: 'All conversations' },
    ...contacts.map((contact) => ({
      value: contact.chat_id,
      label: contactTitle(contact),
      meta: formatNumber(messageCount(contact)),
    })),
  ]
}
