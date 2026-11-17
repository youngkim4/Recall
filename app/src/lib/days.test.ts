import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  addDays,
  addMonths,
  countBetween,
  countOn,
  daysSpanned,
  lastDay,
  lastMonthsRange,
  monthTotals,
  orderDays,
  rangeLabel,
  tidyRange,
  weeksOf,
  yearRange,
  type Daily,
} from './days.ts'

// 30 December 2023 to 3 January 2024
const SHORT: Daily = { start: '2023-12-30', counts: [4, 0, 1, 0, 2] }

// 5 October 2023 to 20 March 2025, one message a day
const LONG: Daily = { start: '2023-10-05', counts: Array.from({ length: 533 }, () => 1) }

// the range separator carries thin spaces in some locales' data
const plain = (text: string) => text.replace(/[\u2009\u202f]/g, ' ')
const label = (since: string, until: string, daily = LONG) => plain(rangeLabel({ since, until }, daily, 'en-US'))

test('days move across months, years and leap days', () => {
  assert.equal(addDays('2024-02-28', 1), '2024-02-29')
  assert.equal(addDays('2023-12-31', 1), '2024-01-01')
  assert.equal(addDays('2024-03-01', -1), '2024-02-29')
})

test('a stretch of time covers every calendar day it touches', () => {
  // ten days and a quarter of an hour apart on the clock, twelve days on the calendar
  assert.equal(daysSpanned('2024-12-06 23:50:00', '2024-12-17 00:05:00'), 12)
  assert.equal(daysSpanned('2024-12-06T08:00:00', '2024-12-06T21:00:00'), 1)
  assert.equal(daysSpanned('2024-12-06 08:00:00', undefined), 0)
  // across the end of a year
  assert.equal(daysSpanned('2023-12-30', '2024-01-03'), 5)
})

test('a month later is the same day, or the last one the month has', () => {
  assert.equal(addMonths('2024-01-15', 1), '2024-02-15')
  assert.equal(addMonths('2024-01-31', 1), '2024-02-29')
  assert.equal(addMonths('2024-03-31', -1), '2024-02-29')
  assert.equal(addMonths('2024-11-30', 3), '2025-02-28')
})

test('a conversation ends on its last counted day', () => {
  assert.equal(lastDay(SHORT), '2024-01-03')
  assert.equal(lastDay({ start: '', counts: [] }), '')
})

test('a day outside the conversation holds nothing', () => {
  assert.equal(countOn(SHORT, '2023-12-30'), 4)
  assert.equal(countOn(SHORT, '2024-01-03'), 2)
  assert.equal(countOn(SHORT, '2023-12-29'), 0)
  assert.equal(countOn(SHORT, '2024-01-04'), 0)
})

test('a range counts both of its ends, and an open side reaches the end', () => {
  assert.equal(countBetween(SHORT, '2023-12-30', '2024-01-01'), 5)
  assert.equal(countBetween(SHORT, '2024-01-01', '2024-01-01'), 1)
  assert.equal(countBetween(SHORT, '', ''), 7)
  assert.equal(countBetween(SHORT, '2024-01-01', ''), 3)
  assert.equal(countBetween(SHORT, '2020-01-01', '2030-01-01'), 7)
  assert.equal(countBetween(SHORT, '2023-12-31', '2023-12-31'), 0)
})

test('a range that the conversation does not reach holds nothing', () => {
  // wholly before its first day
  assert.equal(countBetween(LONG, '2023-01-01', '2023-06-01'), 0)
  assert.equal(countBetween(LONG, '', '2023-06-01'), 0)
  // wholly after its last day
  assert.equal(countBetween(LONG, '2026-01-01', '2026-06-01'), 0)
  assert.equal(countBetween(LONG, '2026-01-01', ''), 0)
  // a range whose ends are the wrong way round
  assert.equal(countBetween(LONG, '2024-06-01', '2024-01-01'), 0)
})

test('two days make a range in either order', () => {
  assert.deepEqual(orderDays('2024-01-03', '2023-12-30'), { since: '2023-12-30', until: '2024-01-03' })
})

test('a range that covers the whole conversation is no range at all', () => {
  assert.deepEqual(tidyRange({ since: '2023-12-30', until: '2024-01-03' }, SHORT), { since: '', until: '' })
  assert.deepEqual(tidyRange({ since: '2023-01-01', until: '2025-01-01' }, SHORT), { since: '', until: '' })
  assert.deepEqual(tidyRange({ since: '2023-12-31', until: '2024-01-03' }, SHORT), {
    since: '2023-12-31',
    until: '2024-01-03',
  })
})

test('a month is laid out in weeks from the day the week starts on', () => {
  // October 2024 starts on a Tuesday
  const sunday = weeksOf('2024-10', 0)
  assert.equal(sunday.length, 5)
  assert.deepEqual(sunday[0], [null, null, '2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04', '2024-10-05'])
  assert.deepEqual(sunday[4], ['2024-10-27', '2024-10-28', '2024-10-29', '2024-10-30', '2024-10-31', null, null])

  const monday = weeksOf('2024-10', 1)
  assert.deepEqual(monday[0], [null, '2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04', '2024-10-05', '2024-10-06'])
})

test('a month can need six weeks or only four', () => {
  // March 2025 starts on a Saturday; February 2026 starts on a Sunday and has 28 days
  assert.equal(weeksOf('2025-03', 0).length, 6)
  assert.equal(weeksOf('2026-02', 0).length, 4)
})

test('months are counted from the days, silent ones included', () => {
  assert.deepEqual(monthTotals(SHORT), [
    { month: '2023-12', total: 4 },
    { month: '2024-01', total: 3 },
  ])
  const quiet: Daily = { start: '2024-01-31', counts: [1, ...Array.from({ length: 29 }, () => 0), 2] }
  assert.deepEqual(monthTotals(quiet), [
    { month: '2024-01', total: 1 },
    { month: '2024-02', total: 0 },
    { month: '2024-03', total: 2 },
  ])
  assert.deepEqual(monthTotals({ start: '', counts: [] }), [])
})

test('the last months count back from the last day', () => {
  assert.deepEqual(lastMonthsRange(LONG, 3), { since: '2024-12-21', until: '2025-03-20' })
  assert.deepEqual(lastMonthsRange(LONG, 12), { since: '2024-03-21', until: '2025-03-20' })
})

test('a conversation no longer than the stretch asked for has no such stretch', () => {
  assert.equal(lastMonthsRange(SHORT, 3), null)
  assert.equal(lastMonthsRange(LONG, 24), null)
})

// two years of days, one message a day, up to and including the day given
const endingOn = (last: string): Daily => ({ start: addDays(last, -729), counts: Array.from({ length: 730 }, () => 1) })

test('the last months from the end of a month are whole months', () => {
  // three months back from 31 May there is no 31 February
  assert.deepEqual(lastMonthsRange(endingOn('2025-05-31'), 3), { since: '2025-03-01', until: '2025-05-31' })
  assert.deepEqual(lastMonthsRange(endingOn('2025-03-31'), 3), { since: '2025-01-01', until: '2025-03-31' })
  assert.deepEqual(lastMonthsRange(endingOn('2025-04-30'), 12), { since: '2024-05-01', until: '2025-04-30' })
  // a year back from a leap day
  assert.deepEqual(lastMonthsRange(endingOn('2024-02-29'), 12), { since: '2023-03-01', until: '2024-02-29' })
  assert.equal(label('2025-03-01', '2025-05-31', endingOn('2025-05-31')), 'Last 3 months')
})

test('a year is the part of it the conversation covers', () => {
  assert.deepEqual(yearRange(LONG, '2023'), { since: '2023-10-05', until: '2023-12-31' })
  assert.deepEqual(yearRange(LONG, '2024'), { since: '2024-01-01', until: '2024-12-31' })
  assert.deepEqual(yearRange(LONG, '2025'), { since: '2025-01-01', until: '2025-03-20' })
  assert.equal(yearRange(LONG, '2019'), null)
})

test('a range is named the shortest true way', () => {
  assert.equal(label('', ''), 'All dates')
  assert.equal(label('2023-10-05', '2025-03-20'), 'All dates')
  assert.equal(label('2024-01-01', '2024-12-31'), '2024')
  // the part of 2023 the conversation covers is all of its 2023
  assert.equal(label('2023-10-05', '2023-12-31'), '2023')
  assert.equal(label('2024-06-01', '2024-06-30'), 'June 2024')
  // October 2023 starts on the 5th here, which is all of its October
  assert.equal(label('2023-10-05', '2023-10-31'), 'October 2023')
  assert.equal(label('2024-03-21', '2025-03-20'), 'Last 12 months')
  assert.equal(label('2024-12-21', '2025-03-20'), 'Last 3 months')
  assert.equal(label('2024-06-10', '2024-06-10'), 'Jun 10, 2024')
  assert.equal(label('2024-06-10', '2024-07-04'), 'Jun 10 – Jul 4, 2024')
  assert.equal(label('2023-12-24', '2024-01-02'), 'Dec 24, 2023 – Jan 2, 2024')
})

test('an open side is named from the end of the conversation', () => {
  assert.equal(label('2025-01-01', ''), '2025')
  assert.equal(label('', '2023-12-31'), '2023')
})

test('a range that reaches past the conversation is named by the part inside it', () => {
  assert.equal(label('2025-03-01', '2025-04-30'), 'March 2025')
  assert.equal(label('2023-09-01', '2023-10-09'), 'Oct 5 – 9, 2023')
})

test('a range with no day of the conversation in it is named by its own days', () => {
  // the conversation ends on 20 March 2025
  assert.equal(label('2025-03-21', '2025-03-25'), 'Mar 21 – 25, 2025')
  assert.equal(label('2020-01-01', '2020-01-01'), 'Jan 1, 2020')
  assert.equal(label('2026-01-01', ''), 'From Jan 1, 2026')
  assert.equal(label('', '2020-01-01'), 'Through Jan 1, 2020')
})

test('a range given back to front is named in order', () => {
  assert.equal(label('2024-07-04', '2024-06-10'), 'Jun 10 – Jul 4, 2024')
  assert.equal(label('2024-12-31', '2024-01-01'), '2024')
  assert.equal(label('2025-03-25', '2025-03-21'), 'Mar 21 – 25, 2025')
})

test('a side that is not a day is an open side', () => {
  assert.equal(label('soon', ''), 'All dates')
  assert.equal(label('2024-13-45', '2024-06-30'), 'Oct 5, 2023 – Jun 30, 2024')
  assert.equal(label('2024-02-30', '2024-02-30'), 'All dates')
})

test('a range outside the conversation is left as it was given', () => {
  const outside = { since: '2025-03-21', until: '2025-03-25' }

  assert.deepEqual(tidyRange(outside, LONG), outside)
  assert.deepEqual(tidyRange({ since: '2024-07-04', until: '2024-06-10' }, LONG), { since: '2024-07-04', until: '2024-06-10' })
})

test('a conversation with no days names every range the same way', () => {
  assert.equal(label('2024-06-10', '2024-07-04', { start: '', counts: [] }), 'All dates')
})
