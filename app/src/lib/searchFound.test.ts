import assert from 'node:assert/strict'
import { test } from 'node:test'
import { keepFound, type Found } from './searchFound.ts'

const found = (key: string, whole: boolean, total: number): Found => ({ key, whole, total, results: [] })

test('the first answer for a search is shown', () => {
  assert.deepEqual(keepFound(null, found('hi', false, 9)), found('hi', false, 9))
})

test('an answer for another search replaces the one shown', () => {
  assert.deepEqual(keepFound(found('h', true, 3), found('hi', false, 9)), found('hi', false, 9))
})

test('whole words replace the matches found as the words were typed', () => {
  assert.deepEqual(keepFound(found('hi', false, 12854), found('hi', true, 410)), found('hi', true, 410))
})

test('whole words that find nothing leave the matches found as the words were typed', () => {
  // stopped at "pizz": "pizza" still shows
  assert.deepEqual(keepFound(found('pizz', false, 40), found('pizz', true, 0)), found('pizz', false, 40))
})

test('whole words that find nothing are shown when nothing else was found', () => {
  assert.deepEqual(keepFound(null, found('zzz', true, 0)), found('zzz', true, 0))
  assert.deepEqual(keepFound(found('zzz', false, 0), found('zzz', true, 0)), found('zzz', true, 0))
})

test('matches found as the words were typed that come late do not replace whole words', () => {
  assert.deepEqual(keepFound(found('hi', true, 410), found('hi', false, 12854)), found('hi', true, 410))
})
