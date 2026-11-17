import type { CardData, CardQuote } from '../../lib/cardCanvas.ts'
import { longDay } from '../../lib/cardData.ts'
import { contactTitle, conversationName, isGroupChat, messageCount } from '../../lib/conversations.ts'
import { sentByMe } from '../../lib/direction.ts'
import { formatLongDate, plural } from '../../lib/format.ts'
import type { Contact, FirstWordsPayload, MemoriesPayload } from '../../types.ts'

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

// a first message on the side that sent it, or on neither when it does not say
const said = (text: string, direction: string): CardQuote => ({ text, fromMe: sentByMe(direction) })

// Everything a card can be made of today: anniversaries that are near, what
// was said on this day in earlier years, and the people written to most.
// Which day "this day" is comes from the server with each moment; this
// computer's clock is not asked.
export function cardMoments(
  memories: MemoriesPayload | null,
  firstWords: FirstWordsPayload | null,
  contacts: Contact[],
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
    const day = item.day || ''
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
