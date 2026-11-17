import assert from 'node:assert/strict'
import { test } from 'node:test'
import { busierThan, daysAround, longDay, spanOf, voicesOf } from './cardData.ts'

const month = (key: string, sent: number, received: number) => ({ key, sent, received, total: sent + received })

test('the two voices over time keep every point, and name the first and the last', () => {
  const chart = voicesOf(
    { unit: 'month', points: [month('2019-03', 4, 9), month('2019-04', 0, 0), month('2019-05', 7, 2)] },
    'en-US',
  )

  assert.deepEqual(chart, {
    unit: 'month',
    points: [
      { sent: 4, received: 9 },
      { sent: 0, received: 0 },
      { sent: 7, received: 2 },
    ],
    first: 'Mar 2019',
    last: 'May 2019',
  })
})

test('a short stretch is by day, and its ends are days', () => {
  const chart = voicesOf({ unit: 'day', points: [month('2024-12-30', 1, 0), month('2025-01-02', 0, 3)] }, 'en-US')

  assert.equal(chart?.first, 'Dec 30')
  assert.equal(chart?.last, 'Jan 2')
})

test('there is no chart of nothing', () => {
  assert.equal(voicesOf(undefined), null)
  assert.equal(voicesOf({ unit: 'month', points: [] }), null)
  assert.equal(voicesOf({ unit: 'month', points: [month('2019-03', 0, 0), month('2019-04', 0, 0)] }), null)
})

test('a count that is not a number is read as none', () => {
  const chart = voicesOf({ unit: 'month', points: [month('2019-03', Number.NaN, 5), month('2019-04', -3, 2)] }, 'en-US')

  assert.deepEqual(chart?.points, [
    { sent: 0, received: 5 },
    { sent: 0, received: 2 },
  ])
})

// 1 October to 30 November 2024, the day's number as its count
const autumn = {
  start: '2024-10-01',
  counts: [...Array.from({ length: 31 }, (_, index) => index + 1), ...Array.from({ length: 30 }, (_, index) => index + 1)],
}

test('a day is set among the days around it', () => {
  const chart = daysAround(autumn, '2024-11-16', 3, 'en-US')

  assert.deepEqual(chart, {
    counts: [13, 14, 15, 16, 17, 18, 19],
    at: 3,
    first: 'Nov 13',
    last: 'Nov 19',
    day: 'Nov 16',
  })
})

test('near an end of the conversation there are fewer days on that side', () => {
  const early = daysAround(autumn, '2024-10-02', 3, 'en-US')
  const late = daysAround(autumn, '2024-11-30', 3, 'en-US')

  assert.deepEqual([early?.counts.length, early?.at, early?.first], [5, 1, 'Oct 1'])
  assert.deepEqual([late?.counts.length, late?.at, late?.last], [4, 3, 'Nov 30'])
})

test('a day the conversation does not hold has no days around it', () => {
  assert.equal(daysAround(autumn, '2024-09-30', 3), null)
  assert.equal(daysAround(autumn, '2024-12-01', 3), null)
  assert.equal(daysAround({ start: '', counts: [] }, '2024-11-16', 3), null)
  assert.equal(daysAround(autumn, 'soon', 3), null)
})

test('a day is busier than the share of the year’s other days that held less', () => {
  // 61 days; 30 October holds 30, and 58 of the other 60 days hold less
  const ranked = busierThan(autumn, '2024-10-30')

  assert.equal(ranked?.year, '2024')
  assert.ok(Math.abs((ranked?.share ?? 0) - 58 / 60) < 1e-9, String(ranked?.share))
})

test('only the days of the same year are counted', () => {
  // 20 December 2023 to 9 February 2024: 12 days of 2023, then 40 of 2024
  const winter = { start: '2023-12-20', counts: [...Array.from({ length: 12 }, () => 99), ...Array.from({ length: 40 }, (_, index) => index)] }

  assert.equal(busierThan(winter, '2024-02-09')?.share, 1)
  // too few days of 2023 to rank one among
  assert.equal(busierThan(winter, '2023-12-25'), null)
})

test('a day with nothing on it, or outside the conversation, is not ranked', () => {
  const quiet = { start: '2024-01-01', counts: Array.from({ length: 60 }, (_, index) => (index === 10 ? 0 : 5)) }

  assert.equal(busierThan(quiet, '2024-01-11'), null)
  assert.equal(busierThan(quiet, '2025-01-11'), null)
})

test('a conversation is named by the months it runs between', () => {
  assert.equal(spanOf('2019-03-04 09:15:00', '2025-04-03 22:01:10', 'en-US'), 'Mar 2019 – Apr 2025')
  assert.equal(spanOf('2019-03-04T09:15:00', '2019-03-28T10:00:00', 'en-US'), 'Mar 2019')
  assert.equal(spanOf('', '2025-04-03 22:01:10', 'en-US'), '')
  assert.equal(spanOf(undefined, undefined), '')
})

test('a day is written out in full, whatever the reader’s time zone', () => {
  assert.equal(longDay('2024-11-16', 'en-US'), 'Saturday, November 16, 2024')
  assert.equal(longDay('not a day', 'en-US'), '')
})
