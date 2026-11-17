import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { MemoriesPayload } from '../types'
import { momentsOf } from './moments.ts'

// invented people and words
const onThisDay = (chatId: string, yearsAgo: number, preview = 'See you at six'): MemoriesPayload['onThisDay'][number] => ({
  chatId,
  name: `Name of ${chatId}`,
  year: 2026 - yearsAgo,
  day: `${2026 - yearsAgo}-11-17`,
  yearsAgo,
  count: 12,
  preview,
  previewFromMe: true,
  previewSender: '',
  previewAt: `${2026 - yearsAgo}-11-17T18:00:00`,
  previewId: `id-${chatId}`,
})

const anniversary = (chatId: string, inDays: number, years = 3): MemoriesPayload['anniversaries'][number] => ({
  chatId,
  name: `Name of ${chatId}`,
  years,
  date: '2023-11-22T09:00:00',
  inDays,
  count: 400,
})

const quiet = (chatId: string): MemoriesPayload['reconnect'][number] => ({
  chatId,
  name: `Name of ${chatId}`,
  count: 5661,
  quietDays: 900,
  lastDate: '2018-10-25T21:30:00',
})

const memories = (parts: Partial<MemoriesPayload>): MemoriesPayload => ({
  onThisDay: [],
  anniversaries: [],
  reconnect: [],
  ...parts,
})

test('the moments are one list: this day first, then the anniversaries, then the quiet', () => {
  const moments = momentsOf(
    memories({ reconnect: [quiet('c')], anniversaries: [anniversary('b', 5)], onThisDay: [onThisDay('a', 2)] }),
  )

  assert.deepEqual(
    moments.map((moment) => [moment.kind, moment.chatId]),
    [
      ['today', 'a'],
      ['anniversary', 'b'],
      ['quiet', 'c'],
    ],
  )
})

test('nothing to show is an empty list', () => {
  assert.deepEqual(momentsOf(memories({})), [])
})

test('each kind shows a few and no more: three of this day, two of each other kind', () => {
  const moments = momentsOf(
    memories({
      onThisDay: ['a', 'b', 'c', 'd'].map((id, index) => onThisDay(id, index + 1)),
      anniversaries: ['e', 'f', 'g'].map((id, index) => anniversary(id, index)),
      reconnect: ['h', 'i', 'j'].map(quiet),
    }),
  )

  assert.deepEqual(
    moments.map((moment) => moment.chatId),
    ['a', 'b', 'c', 'e', 'f', 'h', 'i'],
  )
})

test('a moment of this day says how long ago it was, and carries its quoted message', () => {
  const moments = momentsOf(memories({ onThisDay: [onThisDay('a', 1), onThisDay('b', 9)] }))

  assert.deepEqual(
    moments.map((moment) => moment.when),
    ['A year ago today', '9 years ago today'],
  )
  assert.deepEqual(moments[0].quote, { text: 'See you at six', fromMe: true })
  assert.equal(moments[0].line, '12 messages that day')
  assert.equal(moments[0].day, '2025-11-17')
  assert.equal(moments[0].quotedId, 'id-a')
})

test('a moment of this day with no message to quote has no quote', () => {
  const [moment] = momentsOf(memories({ onThisDay: [onThisDay('a', 2, '')] }))

  assert.equal(moment.quote, undefined)
})

test('an anniversary says how far off it is, and opens on the day it began', () => {
  const moments = momentsOf(memories({ anniversaries: [anniversary('a', 0, 1), anniversary('b', 1), anniversary('c', 6)] }))

  // two are shown
  assert.deepEqual(
    moments.map((moment) => moment.when),
    ['Today', 'Tomorrow'],
  )
  assert.equal(moments[0].line, '1 year since the first message')
  assert.equal(moments[1].line, '3 years since the first message')
  assert.equal(moments[0].day, '2023-11-22')
})

test('a quiet conversation says since when it is quiet, and opens on its last day', () => {
  const [moment] = momentsOf(memories({ reconnect: [quiet('a')] }))

  assert.match(moment.when, /^Quiet since .*2018/)
  assert.equal(moment.line, '5,661 messages')
  assert.equal(moment.day, '2018-10-25')
})

test('every moment has its own key, also when one conversation has several', () => {
  const moments = momentsOf(
    memories({
      onThisDay: [onThisDay('a', 1), onThisDay('a', 2)],
      anniversaries: [anniversary('a', 3)],
      reconnect: [quiet('a')],
    }),
  )
  const keys = moments.map((moment) => moment.key)

  assert.equal(new Set(keys).size, keys.length)
})

test('a quoted message in a group says who wrote it; one with no sender named says nothing', () => {
  const moments = momentsOf(
    memories({ onThisDay: [{ ...onThisDay('a', 1), previewSender: 'Invented Sender' }, onThisDay('b', 2)] }),
  )

  assert.equal(moments[0].quotedFrom, 'Invented Sender')
  assert.equal(moments[1].quotedFrom, undefined)
})

test('a quote whose sender the export does not say takes no side', () => {
  const [moment] = momentsOf(memories({ onThisDay: [{ ...onThisDay('a', 1), previewFromMe: null }] }))

  assert.deepEqual(moment.quote, { text: 'See you at six', fromMe: null })
})

test('a quoted message carries when it was sent, and its id only where there is one', () => {
  const moments = momentsOf(memories({ onThisDay: [onThisDay('a', 1), { ...onThisDay('b', 2), previewId: '' }] }))

  assert.equal(moments[0].at, '2025-11-17T18:00:00')
  assert.equal(moments[1].quotedId, undefined)
})

test('one message is one message', () => {
  const moments = momentsOf(
    memories({ onThisDay: [{ ...onThisDay('a', 1), count: 1 }], reconnect: [{ ...quiet('b'), count: 1 }] }),
  )

  assert.deepEqual(
    moments.map((moment) => moment.line),
    ['1 message that day', '1 message'],
  )
})
