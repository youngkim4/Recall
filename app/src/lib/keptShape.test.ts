import assert from 'node:assert/strict'
import { test } from 'node:test'
import { fakeStorage, noPage, pageWithNoStorage, pageWithStorage } from './fakeStorage.test.ts'
import { readShape, storeShape } from './keptShape.ts'

const KEY = 'recall.shape.moments'

test('with nothing remembered a list has no shape', () => {
  pageWithStorage(fakeStorage())

  assert.equal(readShape('moments'), null)
})

test('the shape a list had comes back', () => {
  pageWithStorage(fakeStorage())

  storeShape('moments', { rows: 7, height: 637 })

  assert.deepEqual(readShape('moments'), { rows: 7, height: 637 })
})

test('each list has its own shape', () => {
  pageWithStorage(fakeStorage())

  storeShape('moments', { rows: 7, height: 637 })
  storeShape('people', { rows: 3, height: 135 })

  assert.deepEqual(readShape('moments'), { rows: 7, height: 637 })
  assert.deepEqual(readShape('people'), { rows: 3, height: 135 })
})

test('a list that had no rows is remembered as empty, which is not the same as not remembered', () => {
  pageWithStorage(fakeStorage())

  storeShape('moments', { rows: 0, height: 0 })

  assert.deepEqual(readShape('moments'), { rows: 0, height: 0 })
})

// a list of lines and bubbles is often half a pixel off a whole number
test('a height is kept to a tenth of a pixel', () => {
  pageWithStorage(fakeStorage())

  storeShape('moments', { rows: 7, height: 637.5 })
  assert.deepEqual(readShape('moments'), { rows: 7, height: 637.5 })

  storeShape('moments', { rows: 5, height: 480.6400146484375 })
  assert.deepEqual(readShape('moments'), { rows: 5, height: 480.6 })
})

test('the same shape is not written a second time', () => {
  const storage = fakeStorage()
  pageWithStorage(storage)

  storeShape('moments', { rows: 7, height: 637.5 })
  storeShape('moments', { rows: 7, height: 637.5 })
  storeShape('moments', { rows: 7, height: 637.52 })

  assert.equal(storage.changes(), 1)
})

test('a shape that could not have been measured is not stored', () => {
  const storage = fakeStorage()
  pageWithStorage(storage)

  storeShape('moments', { rows: -1, height: 100 })
  storeShape('moments', { rows: 2.5, height: 100 })
  storeShape('moments', { rows: 3, height: Number.NaN })
  storeShape('moments', { rows: 3, height: -20 })
  storeShape('moments', { rows: 3, height: Number.POSITIVE_INFINITY })
  storeShape('moments', { rows: 9999, height: 100 })
  storeShape('moments', { rows: 3, height: 1e9 })

  assert.equal(storage.changes(), 0)
  assert.equal(readShape('moments'), null)
})

test('a list that changes is remembered as it is now', () => {
  pageWithStorage(fakeStorage())

  storeShape('moments', { rows: 7, height: 637.5 })
  storeShape('moments', { rows: 5, height: 637.5 })

  assert.deepEqual(readShape('moments'), { rows: 5, height: 637.5 })
})

test('what is stored but is not a shape is read as nothing', () => {
  const stored = [
    'not json',
    '7',
    'null',
    '[]',
    '{}',
    '{"rows":7}',
    '{"rows":"7","height":637}',
    '{"rows":7,"height":-1}',
    '{"rows":7.5,"height":637}',
    '{"rows":7,"height":1e9}',
    '{"rows":9999,"height":637}',
  ]

  for (const value of stored) {
    pageWithStorage(fakeStorage({ [KEY]: value }))
    assert.equal(readShape('moments'), null, value)
  }
})

test('a store that is closed, or no store, or no page at all remembers nothing and breaks nothing', () => {
  pageWithStorage(fakeStorage({}, { broken: true }))
  assert.doesNotThrow(() => storeShape('moments', { rows: 7, height: 637 }))
  assert.equal(readShape('moments'), null)

  // a store with no room left: nothing is written, and nothing is thrown
  const full = pageWithStorage(fakeStorage({}, { roomFor: 4 }))
  assert.doesNotThrow(() => storeShape('moments', { rows: 7, height: 637 }))
  assert.deepEqual(full.dump(), {})

  pageWithNoStorage()
  storeShape('moments', { rows: 7, height: 637 })
  assert.equal(readShape('moments'), null)

  noPage()
  storeShape('moments', { rows: 7, height: 637 })
  assert.equal(readShape('moments'), null)
})
