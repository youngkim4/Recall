import type { Contact, RecentMessage, ReportFile, SearchResult } from '../types'

// a group's id is the word chat and a number; chatterbox@example.com is a person
const GROUP_ID = /^chat\d+$/
const UNKNOWN = 'Unknown conversation'
const UNNAMED_GROUP = 'Unnamed group'

// The one name a conversation goes by, wherever it is shown. The server sends
// the saved name, or for a group nobody named, the people in it. Whatever is
// still unnamed is labelled here and nowhere else. A phone number is never
// shown in full: its last four digits are enough to recognise the thread.
export function conversationName(savedName?: string | null, chatId?: string | null) {
  const name = (savedName || '').trim()
  if (name) return name
  const id = (chatId || '').trim()
  if (!id) return UNKNOWN
  if (GROUP_ID.test(id)) return UNNAMED_GROUP
  if (id.includes('@')) return id
  const digits = id.replace(/\D/g, '')
  // a short code is already its own name
  if (digits.length < 7) return id
  return `Number ending in ${digits.slice(-4)}`
}

// What one person is called in a sentence, as in "You and Avery": the first
// name of someone who is saved. A label that stands in for a name stays
// whole, and so do an address, a number, and the people of a group.
export function shortName(name: string) {
  const whole = name.trim()
  if (whole === UNKNOWN || whole === UNNAMED_GROUP || /[\d@,&]|\sand\s/i.test(whole)) return whole
  return whole.split(/\s+/)[0]
}

export function contactTitle(contact: Contact) {
  return conversationName(contact.display_name || contact.displayName, contact.chat_id)
}

export function reportTitle(report: ReportFile) {
  if (report.displayName) return report.displayName
  return report.contact ? conversationName('', report.contact) : report.name
}

export function isGroupChat(chatId?: string | null) {
  return GROUP_ID.test((chatId || '').trim())
}

export function messageCount(contact: Contact) {
  return Number(contact.message_count || 0)
}

// What each side wrote, as the list says it. Null when the list does not
// say: the Messages database is listed with totals only, and an export with
// no direction counts every message for neither voice. A message that does
// not say who sent it is in the total and in neither voice, so what was
// received is never worked out as the total less what you sent.
export function voicesOf(contact: Contact): { sent: number; received: number } | null {
  if (contact.sent_count === undefined || contact.received_count === undefined) return null
  const total = messageCount(contact)
  const sent = Math.min(total, Math.max(0, Number(contact.sent_count)))
  const received = Math.min(total - sent, Math.max(0, Number(contact.received_count)))
  return sent + received > 0 ? { sent, received } : null
}

export function sentCount(contact: Contact) {
  return Math.min(messageCount(contact), Number(contact.sent_count || 0))
}

// What the other side wrote; nothing when the list does not say.
export function receivedCount(contact: Contact) {
  return voicesOf(contact)?.received ?? 0
}

// The server says true, false or null. A message kept by an older build may
// say it as 1 or as text.
export function isOutbound(message: Pick<RecentMessage, 'isFromMe'>) {
  const value: unknown = message.isFromMe
  return value === true || value === 1 || (typeof value === 'string' && /^(1|true)$/i.test(value.trim()))
}

// Who a found message is from, and where: "You to Sam", "Sam in Family", "Sam".
export function messageSource(message: SearchResult) {
  const conversation = conversationName(message.displayName, message.chatId)
  const inGroup = isGroupChat(message.chatId)
  if (isOutbound(message)) return inGroup ? `You in ${conversation}` : `You to ${conversation}`
  if (!inGroup) return conversation
  // someone in a group who is not in Contacts is known by their number's last digits
  const sender = message.senderName || (message.sender ? conversationName('', message.sender) : '')
  return sender ? `${sender} in ${conversation}` : conversation
}
