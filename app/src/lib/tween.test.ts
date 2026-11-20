import assert from 'node:assert/strict'
import { test } from 'node:test'
import { easeOut, mix, resample } from './tween.ts'

test('a series keeps its values at its own length', () => {
  assert.deepEqual(resample([1, 5, 2], 3), [1, 5, 2])
})

test('a series is stretched or squeezed along its length, by straight lines between its points', () => {
  // three points stretched to five: the new ones fall halfway between
  assert.deepEqual(resample([0, 10, 20], 5), [0, 5, 10, 15, 20])
  // five squeezed to three: the ends and the middle
  assert.deepEqual(resample([0, 5, 10, 15, 20], 3), [0, 10, 20])
})

test('a series of one point, or of none, stands flat', () => {
  assert.deepEqual(resample([7], 3), [7, 7, 7])
  // nothing before: it grows from the baseline
  assert.deepEqual(resample([], 2), [0, 0])
  assert.deepEqual(resample([3, 4], 0), [])
})

test('two series of one length are mixed point by point', () => {
  assert.deepEqual(mix([0, 10], [10, 30], 0), [0, 10])
  assert.deepEqual(mix([0, 10], [10, 30], 0.5), [5, 20])
  assert.deepEqual(mix([0, 10], [10, 30], 1), [10, 30])
})

test('the easing starts fast and settles, and stays inside its ends', () => {
  assert.equal(easeOut(0), 0)
  assert.equal(easeOut(1), 1)
  assert.ok(easeOut(0.25) > 0.5)
  assert.equal(easeOut(-1), 0)
  assert.equal(easeOut(2), 1)
})
