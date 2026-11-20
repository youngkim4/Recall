import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { StatsRow } from './insights.ts'
import { LATELY_DAYS, latelyRows, latelyWindows, sparkAsk } from './lately.ts'

const row = (key: string, total: number, conversations = [key]): StatsRow => ({
  key,
  label: key,
  total,
  sent: 0,
  received: total,
  conversations,
})

test('lately is the last thirty days of the archive, against the thirty before them', () => {
  assert.equal(LATELY_DAYS, 30)
  assert.deepEqual(latelyWindows('2026-11-16'), {
    now: { since: '2026-10-18', until: '2026-11-16' },
    before: { since: '2026-09-18', until: '2026-10-17' },
  })
})

test('each person of this month is set against the month before, through a conversation they share', () => {
  const now = [row('a', 300, ['a', 'a-mail']), row('b', 120), row('c', 50)]
  // a's busiest thread was another one last month: the person is the same
  const before = [row('a-mail', 200, ['a', 'a-mail']), row('b', 240)]
  assert.deepEqual(
    latelyRows(now, before, 3).map((item) => [item.key, item.total, item.change]),
    [
      ['a', 300, 50],
      ['b', 120, -50],
      // nothing last month: no change to name
      ['c', 50, null],
    ],
  )
})

test('only as many as are asked for, the busiest first as the server sends them', () => {
  assert.equal(latelyRows([row('a', 3), row('b', 2), row('c', 1)], [], 2).length, 2)
})

test('the small line of a person is their last eight weeks, every conversation of theirs', () => {
  assert.deepEqual(sparkAsk(row('a', 1, ['a', 'a-mail']), '2026-11-16'), {
    by: 'week',
    contacts: 'a,a-mail',
    since: '2026-09-22',
    until: '2026-11-16',
  })
})
