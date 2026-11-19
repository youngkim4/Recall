import assert from 'node:assert/strict'
import { test } from 'node:test'
import { NO_FILTERS } from './searchFilters.ts'
import { readAsk, readStatsKey, rhythmOf, statsAsk, statsKey } from './insights.ts'

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

test('the rows by weekday and hour fill seven rows of twenty-four hours from Monday', () => {
  const rhythm = rhythmOf([
    { key: '0-09', label: 'Monday 09:00', total: 4, sent: 1, received: 3 },
    { key: '6-23', label: 'Sunday 23:00', total: 2, sent: 2, received: 0 },
    { key: '9-99', label: 'nothing', total: 50, sent: 0, received: 50 },
    { key: 'x', label: 'nothing', total: 50, sent: 0, received: 50 },
  ])
  assert.equal(rhythm.length, 7)
  assert.ok(rhythm.every((day) => day.length === 24))
  assert.equal(rhythm[0][9], 4)
  assert.equal(rhythm[6][23], 2)
  assert.equal(
    rhythm.flat().reduce((sum, count) => sum + count, 0),
    6,
  )
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
