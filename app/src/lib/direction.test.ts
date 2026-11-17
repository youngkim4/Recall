import assert from 'node:assert/strict'
import { test } from 'node:test'
import { sentByMe } from './direction.ts'

test('a first message says which way it went in a word', () => {
  assert.equal(sentByMe('outgoing'), true)
  assert.equal(sentByMe('incoming'), false)
})

test('a message that does not say which way it went belongs to neither side', () => {
  for (const word of ['', undefined, null, 'sideways']) assert.equal(sentByMe(word), null)
})
