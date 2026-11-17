import assert from 'node:assert/strict'
import { test } from 'node:test'
import { byYear, lastDayOf, nextMonth, shiftMonth, type MonthCount } from './months.ts'

// October 2023 to March 2025, with one silent month
const TIMELINE: MonthCount[] = [
  ['2023-10', 40],
  ['2023-11', 10],
  ['2023-12', 0],
  ['2024-01', 5],
  ['2024-12', 5],
  ['2025-01', 20],
  ['2025-02', 30],
  ['2025-03', 7],
].map(([month, total]) => ({ month: String(month), total: Number(total) }))

test('a month moves across the turn of a year', () => {
  assert.equal(shiftMonth('2024-11', 3), '2025-02')
  assert.equal(shiftMonth('2024-01', -1), '2023-12')
  assert.equal(shiftMonth('2024-06', 0), '2024-06')
  assert.equal(shiftMonth('2024-06', -18), '2022-12')
})

test('the last day of a month knows about leap years', () => {
  assert.equal(lastDayOf('2024-02'), '2024-02-29')
  assert.equal(lastDayOf('2025-02'), '2025-02-28')
  assert.equal(lastDayOf('2024-12'), '2024-12-31')
})

test('years are rows of twelve, blank where the conversation has not reached', () => {
  const years = byYear(TIMELINE)

  assert.deepEqual(years.map((row) => row.year), ['2023', '2024', '2025'])
  assert.equal(years[0].slots.length, 12)
  assert.equal(years[0].slots[8], null)
  assert.deepEqual(years[0].slots[9], { month: '2023-10', total: 40 })
  assert.deepEqual(years[2].slots[2], { month: '2025-03', total: 7 })
  assert.equal(years[2].slots[3], null)
})

test('the next month with messages skips silent ones and stops at the ends', () => {
  assert.equal(nextMonth(TIMELINE, '2023-11', 1), '2024-01')
  assert.equal(nextMonth(TIMELINE, '2024-01', -1), '2023-11')
  assert.equal(nextMonth(TIMELINE, '2025-03', 1), '2025-03')
  assert.equal(nextMonth(TIMELINE, '2024-01', 12), '2025-01')
  // nothing a year before October 2023: stay put
  assert.equal(nextMonth(TIMELINE, '2023-10', -12), '2023-10')
})
