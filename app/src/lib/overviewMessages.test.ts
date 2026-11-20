import assert from 'node:assert/strict'
import test from 'node:test'
import { greetingAt, grainOf, stretchAsk, stretchWords } from './overviewMessages.ts'

test('all time asks for every message by month, as Insights first does', () => {
  assert.deepEqual(stretchAsk('all', '2026-11-16'), { by: 'month' })
})

test('the last 30 days end on the archive’s last day and count each day', () => {
  assert.deepEqual(stretchAsk('month', '2026-11-16'), { by: 'day', since: '2026-10-18', until: '2026-11-16' })
  assert.equal(grainOf('month'), 'day')
})

test('the last year is 365 days to the last day, by week', () => {
  assert.deepEqual(stretchAsk('year', '2026-11-16'), { by: 'week', since: '2025-11-17', until: '2026-11-16' })
})

test('with no last day yet, every stretch asks for all time', () => {
  assert.deepEqual(stretchAsk('month', ''), { by: 'month' })
})

test('the words beside the figure name the stretch', () => {
  const all = { conversations: 1404, since: 'June 2015' }
  assert.equal(stretchWords('month', all), 'in the last 30 days')
  assert.equal(stretchWords('year', all), 'in the last year')
  assert.match(stretchWords('all', all), /^in 1.404 conversations since June 2015$/)
  assert.equal(stretchWords('all', { conversations: 1, since: '' }), 'in 1 conversation')
})

test('the greeting follows the hour', () => {
  assert.equal(greetingAt(4), 'Good evening')
  assert.equal(greetingAt(5), 'Good morning')
  assert.equal(greetingAt(11), 'Good morning')
  assert.equal(greetingAt(12), 'Good afternoon')
  assert.equal(greetingAt(17), 'Good afternoon')
  assert.equal(greetingAt(18), 'Good evening')
  assert.equal(greetingAt(23), 'Good evening')
})
