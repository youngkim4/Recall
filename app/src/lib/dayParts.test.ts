import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { DayMessages, SearchResult } from '../types.ts'
import { dayOfMessage, joinPart, partToAsk, quotedIndex } from './dayParts.ts'

const message = (place: number): SearchResult => ({
  messageId: `m${place}`,
  chatId: '+15550000001',
  timestamp: `2024-11-16 09:${String(place % 60).padStart(2, '0')}:00`,
  text: `text ${place}`,
})

// the messages of a day from one place up to another
function part(total: number, from: number, to: number): DayMessages {
  return {
    contact: '+15550000001',
    day: '2024-11-16',
    total,
    offset: from,
    previousDay: null,
    nextDay: null,
    messages: Array.from({ length: to - from }, (_, index) => message(from + index)),
  }
}

const places = (day: DayMessages) => day.messages.map((one) => one.messageId)

test('later asks for what follows the last message shown, and no more than the day holds', () => {
  assert.deepEqual(partToAsk(part(500, 150, 350), false), { offset: 350, limit: 150 })
  assert.deepEqual(partToAsk(part(900, 150, 350), false), { offset: 350, limit: 200 })
})

test('earlier asks for what comes before the first message shown, and stops at it', () => {
  assert.deepEqual(partToAsk(part(500, 250, 450), true), { offset: 50, limit: 200 })
  assert.deepEqual(partToAsk(part(500, 120, 320), true), { offset: 0, limit: 120 })
})

test('a side with nothing more on it is not asked for', () => {
  assert.equal(partToAsk(part(500, 0, 200), true), null)
  assert.equal(partToAsk(part(500, 300, 500), false), null)
})

test('a long day read in parts, both ways, holds each message once and in order', () => {
  const whole = part(500, 0, 500)
  const serve = (ask: { offset: number; limit: number }) => part(500, ask.offset, ask.offset + ask.limit)
  const read = (shown: DayMessages, earlier: boolean): DayMessages => {
    const ask = partToAsk(shown, earlier)
    return ask ? read(joinPart(shown, serve(ask), earlier), earlier) : shown
  }

  const all = read(read(part(500, 249, 349), true), false)

  assert.equal(all.offset, 0)
  assert.deepEqual(places(all), places(whole))
})

test('a part that meets what is shown is joined to it on its own side', () => {
  const shown = part(500, 200, 400)

  const withEarlier = joinPart(shown, part(500, 0, 200), true)
  const withLater = joinPart(shown, part(500, 400, 500), false)

  assert.equal(withEarlier.offset, 0)
  assert.equal(withEarlier.messages.length, 400)
  assert.equal(withLater.offset, 200)
  assert.deepEqual(places(withLater).slice(-2), ['m498', 'm499'])
})

test('a part that no longer meets what is shown is left out', () => {
  const shown = part(500, 200, 400)

  // the same part arriving a second time, on either side
  const again = joinPart(joinPart(shown, part(500, 0, 200), true), part(500, 0, 200), true)
  const twiceLater = joinPart(joinPart(shown, part(500, 400, 500), false), part(500, 400, 500), false)

  assert.equal(again.messages.length, 400)
  assert.equal(twiceLater.messages.length, 300)
  // a part cut short by an older server, which starts before the place asked for
  assert.equal(joinPart(shown, part(500, 300, 500), false), shown)
})

test('a part with nothing in it changes nothing', () => {
  const shown = part(500, 200, 400)

  assert.equal(joinPart(shown, part(500, 400, 400), false), shown)
  assert.equal(joinPart(shown, part(500, 200, 200), true), shown)
})

test('the quoted message is found by its id before its time', () => {
  // three messages sent in the same second
  const sameSecond = [0, 1, 2].map((place) => ({ ...message(place), timestamp: '2024-11-16 09:00:00' }))

  assert.equal(quotedIndex(sameSecond, 'm2', '2024-11-16 09:00:00'), 2)
})

test('with no id, or an id the day does not hold, the first message sent at that time is the one', () => {
  const day = [0, 1, 2].map(message)

  assert.equal(quotedIndex(day, '', '2024-11-16 09:01:00'), 1)
  assert.equal(quotedIndex(day, undefined, '2024-11-16 09:02:00'), 2)
  assert.equal(quotedIndex(day, 'gone', '2024-11-16 09:01:00'), 1)
})

test('a moment that quotes nothing, or a message the day does not hold, picks no message', () => {
  const day = [0, 1, 2].map(message)

  assert.equal(quotedIndex(day, null, null), -1)
  assert.equal(quotedIndex(day, 'gone', '2024-11-16 23:00:00'), -1)
})

test('the day of a message is the first ten letters of its time', () => {
  assert.equal(dayOfMessage('2024-05-01 10:01:00'), '2024-05-01')
  assert.equal(dayOfMessage('2024-05-01'), '2024-05-01')
})

test('a message with no time, or with a time that is not a date, is in no day', () => {
  assert.equal(dayOfMessage(null), null)
  assert.equal(dayOfMessage(undefined), null)
  assert.equal(dayOfMessage(''), null)
  assert.equal(dayOfMessage('yesterday'), null)
  assert.equal(dayOfMessage('NaT'), null)
})
