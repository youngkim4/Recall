import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { AnalysisEvent } from '../types.ts'
import { quotedIndex } from './dayParts.ts'
import { dayOfEvent, dayOfMoment } from './momentDay.ts'
import { readMoments } from './reportMoments.ts'

const event = (fields: Partial<AnalysisEvent> = {}): AnalysisEvent => ({
  date: '2024-05-12',
  title: 'The move',
  detail: 'They talk about the new flat.',
  ...fields,
})

// an event as the new engine writes it
const placed = (fields: Partial<AnalysisEvent> = {}) =>
  event({ chatId: '+15550100199', day: '2024-05-12', at: '2024-05-12 21:14:03', pickedId: 'legacy:4471', ...fields })

test('a moment with its conversation and its day opens that day at the quoted message', () => {
  assert.deepEqual(dayOfEvent(placed()), {
    chatId: '+15550100199',
    day: '2024-05-12',
    at: '2024-05-12 21:14:03',
    pickedId: 'legacy:4471',
  })
})

test('a moment that quotes no message opens its day at the start', () => {
  assert.deepEqual(dayOfEvent(placed({ at: null, pickedId: '' })), {
    chatId: '+15550100199',
    day: '2024-05-12',
    at: null,
    pickedId: undefined,
  })
  assert.equal(dayOfEvent(placed({ at: undefined, pickedId: undefined }))?.at, null)
})

test('a moment of an older report opens nothing', () => {
  assert.equal(dayOfEvent(event()), null)
  assert.equal(dayOfEvent(event({ quote: 'ME: see you there', score: 0.8, category: 'turning_point' })), null)
})

test('a moment needs both its conversation and a day that is a date', () => {
  assert.equal(dayOfEvent(placed({ chatId: '' })), null)
  assert.equal(dayOfEvent(placed({ chatId: undefined })), null)
  assert.equal(dayOfEvent(placed({ day: undefined })), null)
  assert.equal(dayOfEvent(placed({ day: '' })), null)
  assert.equal(dayOfEvent(placed({ day: 'May 2024' })), null)
  assert.equal(dayOfEvent(placed({ chatId: 7 as unknown as string })), null)
  assert.equal(dayOfEvent(undefined), null)
  assert.equal(dayOfEvent(null), null)
})

test('the day is the date alone, whatever follows it', () => {
  assert.equal(dayOfEvent(placed({ day: '2024-05-12T00:00:00' }))?.day, '2024-05-12')
})

test('the id of the quoted message is text and is never read as a number', () => {
  const guid = '8A6F0C2E-1B7D-4E0A-9C55-3F2D1A7B9E10'
  const found = dayOfEvent(placed({ pickedId: guid }))
  const messages = [
    { messageId: 'legacy:12', timestamp: '2024-05-12 21:14:03' },
    { messageId: guid, timestamp: '2024-05-12 21:14:03' },
  ]

  assert.equal(found?.pickedId, guid)
  // the day reader finds the message by it, before it looks at the time
  assert.equal(quotedIndex(messages, found?.pickedId, found?.at), 1)
})

test('each moment of the timeline finds the event it was read from', () => {
  // given out of order, and one with no day: the timeline sorts them
  const events = [
    placed({ date: '2024-09-01', day: '2024-09-01', at: null, pickedId: '', title: 'Autumn' }),
    event({ date: '', title: 'Undated' }),
    placed({ date: '2023-01-15', day: '2023-01-15', title: 'Winter', chatId: 'chat4471' }),
    event({ date: '2024-02-02', title: 'From an older report' }),
  ]

  const days = readMoments(events).map((moment) => [moment.title, dayOfMoment(events, moment.key)])

  assert.deepEqual(days, [
    ['Winter', { chatId: 'chat4471', day: '2023-01-15', at: '2024-05-12 21:14:03', pickedId: 'legacy:4471' }],
    ['From an older report', null],
    ['Autumn', { chatId: '+15550100199', day: '2024-09-01', at: null, pickedId: undefined }],
    ['Undated', null],
  ])
})

test('a key that names no event opens nothing', () => {
  const events = [placed()]

  assert.equal(dayOfMoment(events, 'moment-1'), null)
  assert.equal(dayOfMoment(events, 'moment--1'), null)
  assert.equal(dayOfMoment(events, 'moment-x'), null)
  assert.equal(dayOfMoment(events, ''), null)
  assert.equal(dayOfMoment([], 'moment-0'), null)
})
