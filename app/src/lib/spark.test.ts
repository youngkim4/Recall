import assert from 'node:assert/strict'
import { test } from 'node:test'
import { topShares } from './spark.ts'

test('the largest parts are given as shares of the whole, largest first', () => {
  assert.deepEqual(topShares([10, 50, 5, 30], 100, 3), [0.5, 0.3, 0.1])
})

test('shares stop at the parts there are, and at nothing when the whole is nothing', () => {
  assert.deepEqual(topShares([10, 50], 100, 5), [0.5, 0.1])
  assert.deepEqual(topShares([10, 50], 0, 5), [])
})

test('a part cannot be more than what is left of the whole', () => {
  // the list can be newer than the total it is set against
  assert.deepEqual(topShares([80, 70], 100, 2), [0.8, 0.2])
})
