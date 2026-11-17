import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { FirstWordsPayload, MemoriesPayload } from '../../types.ts'
import { cardMoments } from './moments.ts'

const AVERY = '+15550000001'

const onThisDay = (day: string): MemoriesPayload => ({
  onThisDay: [
    {
      chatId: AVERY,
      name: 'Avery Stone',
      year: Number(day.slice(0, 4)),
      day,
      yearsAgo: 2,
      count: 12,
      preview: 'a line worth quoting back',
      previewFromMe: false,
      previewSender: '',
      previewAt: `${day} 21:15:00`,
    },
  ],
  anniversaries: [],
  reconnect: [],
})

const anniversary: MemoriesPayload = {
  onThisDay: [],
  anniversaries: [{ chatId: AVERY, name: 'Avery Stone', years: 3, date: '2021-11-20 08:00:00', inDays: 4, count: 900 }],
  reconnect: [],
}

const firstWords = (direction: string, replyDirection: string): FirstWordsPayload => ({
  entries: [
    {
      person: 'Avery Stone',
      chatId: AVERY,
      timestamp: '2021-11-20 08:00:00',
      yearsAgo: 3,
      direction,
      text: 'hello there',
      reply: { text: 'hello back', direction: replyDirection, timestamp: '2021-11-20 08:01:00' },
      messageCount: 900,
    },
  ],
  signature: 'test',
  totals: { messages: 900, people: 1 },
})

test('a card of this day carries the day the server named', () => {
  // 29 February has no "same day" in most years: only the server's own day says it
  const [card] = cardMoments(onThisDay('2024-02-29'), null, [])

  assert.equal(card.data.kind, 'onthisday')
  assert.equal(card.data.kind === 'onthisday' && card.data.day, '2024-02-29')
  assert.match(card.data.kind === 'onthisday' ? (card.data.date ?? '') : '', /2024/)
})

test('the first messages on an anniversary card stand on the side that sent them', () => {
  const [card] = cardMoments(anniversary, firstWords('outgoing', 'incoming'), [])

  assert.equal(card.data.kind === 'anniversary' && card.data.quote?.fromMe, true)
  assert.equal(card.data.kind === 'anniversary' && card.data.reply?.fromMe, false)
})

test('a first message that does not say who sent it stands on neither side', () => {
  const [card] = cardMoments(anniversary, firstWords('', ''), [])

  assert.equal(card.data.kind === 'anniversary' && card.data.quote?.fromMe, null)
  assert.equal(card.data.kind === 'anniversary' && card.data.reply?.fromMe, null)
})
