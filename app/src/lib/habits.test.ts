import assert from 'node:assert/strict'
import { test } from 'node:test'
import { busiestHour, formatLength, formatSpan, formatWait, weekGrid } from './habits.ts'
import type { StatsRow } from './insights.ts'

const cell = (key: string, total: number): StatsRow => ({ key, label: key, total, sent: 0, received: total })

test('a wait is said in the largest unit that reads well', () => {
  assert.equal(formatWait(null), '—')
  assert.equal(formatWait(0), '0 sec')
  assert.equal(formatWait(42), '42 sec')
  // 950 seconds is 15.8 minutes
  assert.equal(formatWait(950), '16 min')
  assert.equal(formatWait(3599), '60 min')
  assert.equal(formatWait(3600), '1 hr')
  assert.equal(formatWait(5400), '1.5 hr')
  assert.equal(formatWait(36 * 3600), '36 hr')
})

test('a span of days names its months, once when they are one', () => {
  assert.equal(formatSpan('2022-07-09', '2025-08-03'), 'Jul 2022 to Aug 2025')
  assert.equal(formatSpan('2024-03-08', '2024-03-11'), 'Mar 2024')
})

test('a length is whole characters, or a dash with no text', () => {
  assert.equal(formatLength(67.21), '67')
  assert.equal(formatLength(null), '—')
})

test('the busiest hour of the week, the first of a tie', () => {
  const rows = [cell('0-17', 40), cell('2-09', 90), cell('4-21', 90)]
  assert.deepEqual(busiestHour(rows), { weekday: 2, hour: 9, total: 90 })
  assert.equal(busiestHour([cell('0-00', 0)]), null)
  assert.equal(busiestHour([]), null)
})

test('the grid of the week has a row a day in the given order and a cell an hour', () => {
  // Monday 17:00 holds 40, Wednesday 09:00 holds 80: the server counts weekdays from Monday
  const grid = weekGrid([cell('0-17', 40), cell('2-09', 80)], [1, 2, 3, 4, 5, 6, 0])
  assert.equal(grid.length, 7)
  assert.ok(grid.every((row) => row.cells.length === 24))
  // the order is of Date: 1 is Monday
  assert.deepEqual(
    grid.map((row) => row.weekday),
    [1, 2, 3, 4, 5, 6, 0],
  )
  assert.deepEqual(grid[0].cells[17], { hour: 17, total: 40, level: 0.5 })
  assert.deepEqual(grid[2].cells[9], { hour: 9, total: 80, level: 1 })
  assert.deepEqual(grid[6].cells[0], { hour: 0, total: 0, level: 0 })
})

test('a week with no message is all at level 0', () => {
  const grid = weekGrid([], [0, 1, 2, 3, 4, 5, 6])
  assert.ok(grid.every((row) => row.cells.every((each) => each.level === 0)))
})
