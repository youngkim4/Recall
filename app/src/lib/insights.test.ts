import assert from 'node:assert/strict'
import { test } from 'node:test'
import { NO_FILTERS } from './searchFilters.ts'
import { groupKind, querySentence, readAsk, readStatsKey, statsAsk, statsKey } from './insights.ts'

test('a count asks with only what is set, its words, and the grouping', () => {
  assert.deepEqual(statsAsk(NO_FILTERS, '', 'month'), { by: 'month' })
  assert.deepEqual(
    statsAsk({ ...NO_FILTERS, from: 'me', since: '2025-01-01', order: 'oldest' }, '  love  ', 'person', 10),
    {
      by: 'person',
      query: 'love',
      from: 'me',
      since: '2025-01-01',
      limit: 10,
    },
  )
})

test('habits are asked for only by name, since they read every message of the set', () => {
  assert.deepEqual(statsAsk(NO_FILTERS, '', 'person', 10, true), { by: 'person', limit: 10, habits: true })
  assert.equal('habits' in statsAsk(NO_FILTERS, '', 'person', 10), false)
})

test('the sort of search is never sent with a count', () => {
  assert.equal('order' in statsAsk({ ...NO_FILTERS, order: 'relevance' }, 'hi', 'week'), false)
})

test('an ask travels as one key and comes back the same', () => {
  const ask = statsAsk({ ...NO_FILTERS, kind: 'group' }, 'pizza', 'weekday_hour')
  assert.deepEqual(readAsk(JSON.stringify(ask)), ask)
  assert.equal(readAsk('not json'), null)
  assert.equal(readAsk('[1,2]'), null)
})

test('a key holds the path of the archive and the ask, and gives both back', () => {
  const ask = statsAsk(NO_FILTERS, 'hi', 'month')
  assert.deepEqual(readStatsKey(statsKey('/a/b.db', ask)), {
    messagesPath: '/a/b.db',
    ask,
  })
  assert.equal(statsKey('', ask), '')
  assert.equal(readStatsKey('no line break'), null)
  assert.equal(readStatsKey('\n{}'), null)
})

test('the question reads back as one sentence, naming only what is set', () => {
  assert.equal(querySentence({ words: '', filters: NO_FILTERS, by: 'month' }), 'All messages, by month')
  assert.equal(
    querySentence({ words: ' love ', filters: { ...NO_FILTERS, from: 'me', kind: 'group' }, by: 'weekday' }),
    'Messages with “love” from you in group chats, by weekday',
  )
  assert.equal(
    querySentence({
      words: '',
      filters: { ...NO_FILTERS, contact: 'chat-1', attachments: 'image' },
      by: 'hour',
      where: 'Sam',
    }),
    'All messages with Sam with a photo, by hour',
  )
  assert.equal(
    querySentence({
      words: 'hi',
      filters: { ...NO_FILTERS, from: 'them', since: '2025-01-01', until: '2025-03-31' },
      by: 'person',
      range: 'Jan 1 to Mar 31, 2025',
    }),
    'Messages with “hi” from others, Jan 1 to Mar 31, 2025, by person',
  )
})

test('a grouping is over time, over the week, or by who', () => {
  assert.equal(groupKind('day'), 'time')
  assert.equal(groupKind('month'), 'time')
  assert.equal(groupKind('weekday'), 'cycle')
  assert.equal(groupKind('hour'), 'cycle')
  assert.equal(groupKind('person'), 'who')
  assert.equal(groupKind('conversation'), 'who')
})
