import assert from 'node:assert/strict'
import { test } from 'node:test'
import { fitSize } from './fitText.ts'

const RANGE = { min: 28, max: 64 }

test('a line that fits at the largest size is set at it', () => {
  assert.deepEqual(fitSize({ room: 2000, widthAt: 1000, at: 52, ...RANGE }), { size: 64, wraps: false })
})

test('a line is scaled so it fills the room it has', () => {
  // 1380px at 52px, in 1000px of room, a hair under it: 52 * 1000 * 0.995 / 1380
  const { size, wraps } = fitSize({ room: 1000, widthAt: 1380, at: 52, ...RANGE })
  assert.equal(wraps, false)
  assert.ok(size > 37.4 && size < 37.5, String(size))
})

test('a line too long for the smallest size is set at it and wraps', () => {
  assert.deepEqual(fitSize({ room: 300, widthAt: 1380, at: 52, ...RANGE }), { size: 28, wraps: true })
})

test('nothing measured yet keeps the largest size', () => {
  assert.deepEqual(fitSize({ room: 0, widthAt: 0, at: 52, ...RANGE }), { size: 64, wraps: false })
})
