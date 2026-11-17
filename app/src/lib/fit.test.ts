import assert from 'node:assert/strict'
import { test } from 'node:test'
import { rowsThatFit } from './fit.ts'

test('a box holds as many whole rows as its height has room for', () => {
  assert.equal(rowsThatFit(450, 45, 5), 10)
  assert.equal(rowsThatFit(493, 45, 5), 10)
  assert.equal(rowsThatFit(495, 45, 5), 11)
})

test('the first row has no line over it, so a list measures one pixel short of its rows', () => {
  assert.equal(rowsThatFit(449, 45, 5), 10)
  assert.equal(rowsThatFit(494, 45, 5), 11)
  assert.equal(rowsThatFit(448.4, 45, 5), 9)
})

test('a short box still shows the fewest rows asked for', () => {
  assert.equal(rowsThatFit(100, 45, 5), 5)
  assert.equal(rowsThatFit(0, 45, 5), 5)
})

test('a row with no height asks for the fewest, not for endless rows', () => {
  assert.equal(rowsThatFit(500, 0, 5), 5)
  assert.equal(rowsThatFit(Number.NaN, 45, 5), 5)
})
