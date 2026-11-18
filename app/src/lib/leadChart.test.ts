import assert from 'node:assert/strict'
import { test } from 'node:test'
import { peakAt, yearTotals } from './leadChart.ts'

const month = (key: string, total: number) => ({ key, total, sent: 0, received: 0 })

test('the months of a year are added up, and the years come oldest first', () => {
  const months = [month('2016-01', 5), month('2015-11', 2), month('2015-12', 3), month('2016-02', 1)]
  assert.deepEqual(yearTotals(months), [
    { year: '2015', total: 5 },
    { year: '2016', total: 6 },
  ])
  assert.deepEqual(yearTotals([]), [])
})

test('the peak is the first largest value, and there is none in nothing', () => {
  assert.equal(peakAt([1, 9, 9, 2]), 1)
  assert.equal(peakAt([]), -1)
})
