import assert from 'node:assert/strict'
import { test } from 'node:test'
import { rowsThatFit } from './fit.ts'

test('a box that is a whole number of rows tall holds that many', () => {
  assert.equal(rowsThatFit(450, 45, 5), 10)
  assert.equal(rowsThatFit(495, 45, 5), 11)
})

// The rows of a list share the height of its box. A list beside another list
// must not have rows that are plainly taller than its neighbor's.
test('one more row is shown when each row is then a twentieth shorter at most', () => {
  // 13 rows of 44.7 each, and not 12 rows of 48.5
  assert.equal(rowsThatFit(581.5, 45, 5), 13)
  // 11 rows of 43.2 each
  assert.equal(rowsThatFit(475, 45, 5), 11)
})

test('a row is not made more than a twentieth shorter for the sake of one more', () => {
  // 11 rows would be 42.7 each: 10 rows of 46.9 are shown
  assert.equal(rowsThatFit(469.5, 45, 5), 10)
  // 6 rows would be 41 each
  assert.equal(rowsThatFit(246, 45, 1), 5)
  // 2 rows would be 33 each
  assert.equal(rowsThatFit(66, 45, 1), 1)
})

test('a row that would be exactly a twentieth shorter is shown, and one a little shorter than that is not', () => {
  // 10 rows of 38 in a box of 380: a row of 40 less a twentieth
  assert.equal(rowsThatFit(380, 40, 1), 10)
  // 10 rows of 37.9
  assert.equal(rowsThatFit(379, 40, 1), 9)
})

test('a height measured in fractions of a pixel does not lose a row', () => {
  assert.equal(rowsThatFit(449.4, 45, 5), 10)
  assert.equal(rowsThatFit(494.6, 45, 5), 11)
})

test('a short box still shows the fewest rows asked for', () => {
  assert.equal(rowsThatFit(100, 45, 5), 5)
  assert.equal(rowsThatFit(0, 45, 5), 5)
  assert.equal(rowsThatFit(-20, 45, 5), 5)
})

test('a row with no height asks for the fewest, not for endless rows', () => {
  assert.equal(rowsThatFit(500, 0, 5), 5)
  assert.equal(rowsThatFit(Number.NaN, 45, 5), 5)
})
