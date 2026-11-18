import assert from 'node:assert/strict'
import { test } from 'node:test'
import { splitByRanges, splitByTerm } from './highlight.ts'

test('the ranges the server matched are picked out, and nothing else', () => {
  // "Hi" as a whole word: "this" and "which" are not marked
  assert.deepEqual(splitByRanges('Hi, this is which', [[0, 2]]), [
    { text: 'Hi', match: true },
    { text: ', this is which', match: false },
  ])
  assert.deepEqual(splitByRanges('one two one', [[0, 3], [8, 11]]), [
    { text: 'one', match: true },
    { text: ' two ', match: false },
    { text: 'one', match: true },
  ])
})

test('ranges out of order, overlapping, empty or outside the text are made safe', () => {
  assert.deepEqual(splitByRanges('abcdef', [[4, 9], [0, 2], [1, 3], [5, 5], [-3, -1], [12, 14]]), [
    { text: 'abc', match: true },
    { text: 'd', match: false },
    { text: 'ef', match: true },
  ])
})

test('no ranges, or none that are numbers, leave the text whole', () => {
  assert.deepEqual(splitByRanges('nothing here', []), [{ text: 'nothing here', match: false }])
  const broken = [['a', 'b']] as unknown as Array<[number, number]>
  assert.deepEqual(splitByRanges('nothing here', broken), [{ text: 'nothing here', match: false }])
  assert.deepEqual(splitByRanges('', [[0, 2]]), [])
})

test('ranges count in UTF-16 units, as the server sends them', () => {
  // the emoji is two units: "hi" starts at 3
  assert.deepEqual(splitByRanges('\u{1F600} hi', [[3, 5]]), [
    { text: '\u{1F600} ', match: false },
    { text: 'hi', match: true },
  ])
})

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
