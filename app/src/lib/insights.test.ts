import assert from 'node:assert/strict'
import { test } from 'node:test'
import { NO_FILTERS } from './searchFilters.ts'
import { busiest, groupKind, querySentence, readAsk, readStatsKey, statsAsk, statsKey } from './insights.ts'

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
    querySentence({ words: '', filters: { ...NO_FILTERS, contact: 'chat-1', attachments: 'image' }, by: 'hour', where: 'Sam' }),
    'All messages with Sam with a photo, by hour',
  )
  assert.equal(
    querySentence({ words: 'hi', filters: { ...NO_FILTERS, from: 'them', since: '2025-01-01', until: '2025-03-31' }, by: 'person', range: 'Jan 1 to Mar 31, 2025' }),
    'Messages with “hi” from others, Jan 1 to Mar 31, 2025, by person',
  )
})

test('the busiest rows come first, the most in each, a tie in order of time', () => {
  const rows = [
    { key: '2025-01', label: '2025-01', total: 3, sent: 1, received: 2 },
    { key: '2025-02', label: '2025-02', total: 9, sent: 4, received: 5 },
    { key: '2025-03', label: '2025-03', total: 3, sent: 3, received: 0 },
    { key: '2025-04', label: '2025-04', total: 0, sent: 0, received: 0 },
  ]
  assert.deepEqual(
    busiest(rows, 3).map((row) => row.key),
    ['2025-02', '2025-01', '2025-03'],
  )
  // an empty row is never among the busiest
  assert.deepEqual(busiest(rows, 9).length, 3)
})

test('a grouping is over time, over the week, or by who', () => {
  assert.equal(groupKind('day'), 'time')
  assert.equal(groupKind('month'), 'time')
  assert.equal(groupKind('weekday'), 'cycle')
  assert.equal(groupKind('hour'), 'cycle')
  assert.equal(groupKind('person'), 'who')
  assert.equal(groupKind('conversation'), 'who')
})
