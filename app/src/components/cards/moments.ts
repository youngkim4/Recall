import type { CardData, CardQuote } from '../../lib/cardCanvas'
import { longDay } from '../../lib/cardData'
import { contactTitle, conversationName, isGroupChat, messageCount } from '../../lib/conversations'
import { formatLongDate, plural } from '../../lib/format'
import type { Contact, FirstWordsPayload, MemoriesPayload } from '../../types'

export type CardMoment = {
  key: string
  group: 'Anniversaries' | 'On this day' | 'You and them'
  label: string
  detail: string
  // the conversation the card is of; its own figures go onto the card
  chatId: string
  data: CardData
}

// how many conversations are offered for a card of two people
const PEOPLE = 8

const pad = (value: number) => String(value).padStart(2, '0')
const said = (text: string, direction: string): CardQuote => ({ text, fromMe: direction === 'outgoing' })

// today's day of the year in another year, as "YYYY-MM-DD"; '' in a year that has no such day
function sameDayIn(year: number, now: Date) {
  const isDay = new Date(Date.UTC(year, now.getMonth(), now.getDate())).getUTCDate() === now.getDate()
  return isDay ? `${year}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}` : ''
}

// Everything a card can be made of today: anniversaries that are near, what
// was said on this day in earlier years, and the people written to most.
export function cardMoments(
  memories: MemoriesPayload | null,
  firstWords: FirstWordsPayload | null,
  contacts: Contact[],
  now: Date,
): CardMoment[] {
  const firstMessages = new Map((firstWords?.entries ?? []).map((entry) => [entry.chatId, entry]))

  const anniversaries = (memories?.anniversaries ?? []).map((item): CardMoment => {
    const name = conversationName(item.name, item.chatId)
    const first = firstMessages.get(item.chatId)
    return {
      key: `anniversary-${item.chatId}`,
      group: 'Anniversaries',
      label: name,
      detail: `${plural(item.years, 'year')}, ${formatLongDate(item.date)}`,
      chatId: item.chatId,
      data: {
        kind: 'anniversary',
        name,
        group: isGroupChat(item.chatId),
        years: item.years,
        sinceDate: formatLongDate(item.date),
        messageCount: item.count,
        quote: first ? said(first.text, first.direction) : null,
        reply: first?.reply ? said(first.reply.text, first.reply.direction) : null,
      },
    }
  })

  const onThisDay = (memories?.onThisDay ?? []).map((item): CardMoment => {
    const name = conversationName(item.name, item.chatId)
    const day = sameDayIn(item.year, now)
    return {
      key: `day-${item.chatId}-${item.year}`,
      group: 'On this day',
      label: name,
      detail: `${item.year}, ${plural(item.count, 'message')}`,
      chatId: item.chatId,
      data: {
        kind: 'onthisday',
        name,
        yearsAgo: item.yearsAgo,
        count: item.count,
        day: day || undefined,
        date: day ? longDay(day) : undefined,
        quote: item.preview ? { text: item.preview, fromMe: item.previewFromMe } : null,
      },
    }
  })

  const people = contacts
    .filter((contact) => !isGroupChat(contact.chat_id) && messageCount(contact) > 0)
    .slice(0, PEOPLE)
    .map((contact): CardMoment => {
      const name = contactTitle(contact)
      return {
        key: `duo-${contact.chat_id}`,
        group: 'You and them',
        label: name,
        detail: plural(messageCount(contact), 'message'),
        chatId: contact.chat_id,
        data: { kind: 'duo', name, totalMessages: messageCount(contact) },
      }
    })

  return [...anniversaries, ...onThisDay, ...people]
}

// what the box that asks a message onto the card says, or '' when the card has no message to show
export function quoteLabel(data: CardData) {
  if (data.kind === 'duo' || !data.quote?.text.trim()) return ''
  if (data.kind === 'onthisday') return 'Include the message'
  return data.reply?.text.trim() ? 'Include the first messages' : 'Include the first message'
}
