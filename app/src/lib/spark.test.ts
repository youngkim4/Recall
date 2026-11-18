import assert from 'node:assert/strict'
import { test } from 'node:test'
import { sparkArea, sparkLine, topShares } from './spark.ts'

test('the line runs the whole width, with the largest value at the top', () => {
  assert.equal(sparkLine([0, 5, 10], 100, 20), 'M0 20L50 10L100 0')
})

test('a line of equal values lies along the middle, since none is larger than another', () => {
  assert.equal(sparkLine([4, 4, 4], 100, 20), 'M0 10L50 10L100 10')
})

test('nothing above zero is a line along the floor', () => {
  assert.equal(sparkLine([0, 0], 100, 20), 'M0 20L100 20')
})

test('one value is a level line, and none is no line', () => {
  assert.equal(sparkLine([7], 100, 20), 'M0 10L100 10')
  assert.equal(sparkLine([], 100, 20), '')
})

test('the area is the line closed down to the floor', () => {
  assert.equal(sparkArea([0, 10], 100, 20), 'M0 20L100 0L100 20L0 20Z')
  assert.equal(sparkArea([], 100, 20), '')
})

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
