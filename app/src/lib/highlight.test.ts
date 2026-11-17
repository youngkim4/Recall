import assert from 'node:assert/strict'
import { test } from 'node:test'
import { splitByTerm } from './highlight.ts'

test('the words searched for are picked out wherever they are, whatever their case', () => {
  assert.deepEqual(splitByTerm('Pizza night, more pizza', 'pizza'), [
    { text: 'Pizza', match: true },
    { text: ' night, more ', match: false },
    { text: 'pizza', match: true },
  ])
})

test('text with no match comes back whole', () => {
  assert.deepEqual(splitByTerm('nothing here', 'pizza'), [{ text: 'nothing here', match: false }])
  assert.deepEqual(splitByTerm('nothing here', '   '), [{ text: 'nothing here', match: false }])
  assert.deepEqual(splitByTerm('', 'pizza'), [])
})

test('characters that mean something in a pattern are matched as themselves', () => {
  assert.deepEqual(splitByTerm('cost (about $5.00)?', '$5.00)'), [
    { text: 'cost (about ', match: false },
    { text: '$5.00)', match: true },
    { text: '?', match: false },
  ])
})

test('the term is trimmed the way the search trims it', () => {
  assert.deepEqual(splitByTerm('see you soon', '  you '), [
    { text: 'see ', match: false },
    { text: 'you', match: true },
    { text: ' soon', match: false },
  ])
})
