import assert from 'node:assert/strict'
import { test } from 'node:test'
import { axisLabels, pointName, roundUp, type ActivityPoint } from './activity.ts'

const point = (key: string): ActivityPoint => ({ key, total: 1, sent: 1, received: 0 })
const months = (first: string, count: number) =>
  Array.from({ length: count }, (_, index) => {
    const moved = Number(first.slice(0, 4)) * 12 + Number(first.slice(5, 7)) - 1 + index
    return point(`${Math.floor(moved / 12)}-${String((moved % 12) + 1).padStart(2, '0')}`)
  })
const days = (first: string, count: number) =>
  Array.from({ length: count }, (_, index) =>
    point(new Date(Date.UTC(Number(first.slice(0, 4)), Number(first.slice(5, 7)) - 1, Number(first.slice(8, 10)) + index)).toISOString().slice(0, 10)),
  )
const texts = (points: ActivityPoint[], unit: 'day' | 'month') => axisLabels(points, unit, 'en-US').map((label) => label.text)

test('the top of a scale is a round number just above the tallest bar', () => {
  assert.equal(roundUp(237), 250)
  assert.equal(roundUp(154), 200)
  assert.equal(roundUp(9_800), 10_000)
  assert.equal(roundUp(1), 1)
  assert.equal(roundUp(0), 0)
})

test('a scale stops at the nearest of its round steps', () => {
  // 1.2, 2.5 and 6 are steps too, so a bar a little over 1, 2 or 5 does not get a scale twice its height
  assert.equal(roundUp(11), 12)
  assert.equal(roundUp(1_150), 1_200)
  assert.equal(roundUp(21), 25)
  assert.equal(roundUp(2_010), 2_500)
  assert.equal(roundUp(52), 60)
  assert.equal(roundUp(5_100), 6_000)
  assert.deepEqual([13, 26, 31, 41, 61, 81].map(roundUp), [15, 30, 40, 50, 80, 100])
})

test('a bar that is already at a round step is the top of its scale', () => {
  const steps = [12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 1_200, 2_500, 6_000, 120_000]

  assert.deepEqual(steps.map(roundUp), steps)
  assert.deepEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(roundUp), [1, 2, 3, 4, 5, 6, 8, 8, 10, 10])
})

test('a scale is never under its tallest bar, nor more than a third over it', () => {
  Array.from({ length: 30_000 }, (_, index) => index + 1).forEach((value) => {
    const top = roundUp(value)

    assert.ok(top >= value, `${top} is under ${value}`)
    assert.ok(top <= Math.ceil((value * 4) / 3), `${top} is far over ${value}`)
  })
})

test('a few months are each named', () => {
  assert.deepEqual(texts(months('2024-10', 4), 'month'), ['Oct', 'Nov', 'Dec', 'Jan'])
})

test('twelve months are each named, and thirteen go by their years', () => {
  assert.deepEqual(texts(months('2024-01', 12), 'month'), [
    'Jan',
    'Feb',
    'Mar',
    'Apr',
    'May',
    'Jun',
    'Jul',
    'Aug',
    'Sep',
    'Oct',
    'Nov',
    'Dec',
  ])
  assert.deepEqual(texts(months('2024-01', 13), 'month'), ['2024', '2025'])
})

test('many months are named by their years', () => {
  const labels = axisLabels(months('2022-06', 30), 'month', 'en-US')

  assert.deepEqual(labels.map((label) => label.text), ['2022', '2023', '2024'])
  // the opening year starts on the chart's edge; the others sit under their January
  assert.equal(labels[0].atEdge, true)
  assert.equal(labels[1].index, 7)
})

test('an opening year too close to the next one is left out', () => {
  assert.deepEqual(texts(months('2022-11', 30), 'month'), ['2023', '2024', '2025'])
})

test('a few days are each numbered, with the month on the first', () => {
  assert.deepEqual(texts(days('2024-12-06', 5), 'day'), ['Dec 6', '7', '8', '9', '10'])
})

test('ten days are each numbered, and eleven are numbered at steps', () => {
  const ten = axisLabels(days('2024-12-06', 10), 'day', 'en-US')
  const eleven = axisLabels(days('2024-12-06', 11), 'day', 'en-US')

  assert.deepEqual(ten.map((label) => label.text), ['Dec 6', '7', '8', '9', '10', '11', '12', '13', '14', '15'])
  // each day has room for its label, so the first sits under its own bar
  assert.equal(ten[0].atEdge, undefined)
  assert.deepEqual(eleven.map((label) => label.text), ['Dec 6', '8', '10', '12', '14', '16'])
  assert.deepEqual(eleven.map((label) => label.index), [0, 2, 4, 6, 8, 10])
  assert.equal(eleven[0].atEdge, true)
})

test('many days are numbered at even steps, and a new month is named', () => {
  assert.deepEqual(texts(days('2024-12-06', 31), 'day'), ['Dec 6', '10', '14', '18', '22', '26', '30', 'Jan 3'])
})

test('a point is named in full for the readout', () => {
  assert.equal(pointName('2024-03', 'month', 'en-US'), 'March 2024')
  assert.equal(pointName('2024-12-08', 'day', 'en-US'), 'Sun, Dec 8, 2024')
})

test('a week is named by its Monday, and its axis is labelled like days', () => {
  assert.equal(pointName('2025-03-03', 'week', 'en-US'), 'Week of Mar 3, 2025')
  const weeks: ActivityPoint[] = ['2025-03-03', '2025-03-10', '2025-03-17'].map((key) => ({ key, total: 1, sent: 1, received: 0 }))
  assert.deepEqual(
    axisLabels(weeks, 'week', 'en-US').map((label) => label.text),
    axisLabels(weeks, 'day', 'en-US').map((label) => label.text),
  )
})
