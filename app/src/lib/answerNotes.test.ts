import assert from 'node:assert/strict'
import { test } from 'node:test'
import { answerNotes } from './answerNotes.ts'

const answer = { id: 'assistant-1-a', role: 'assistant' as const, content: 'An answer.' }
const response = { question: 'q', terms: [], answer: 'An answer.', citations: [] }

test('an answer with nothing to say of it has no note', () => {
  assert.deepEqual(answerNotes(answer), [])
  assert.deepEqual(answerNotes({ ...answer, response }), [])
  assert.deepEqual(answerNotes({ ...answer, response: { ...response, unverifiedQuotes: [], truncated: false } }), [])
})

test('a quote that the archive does not hold is said, with how many', () => {
  const one = { ...answer, response: { ...response, unverifiedQuotes: ['see you at eight'] } }
  const two = { ...answer, response: { ...response, unverifiedQuotes: ['a', 'b'] } }
  assert.deepEqual(answerNotes(one), ['1 quote in this answer is not in your messages.'])
  assert.deepEqual(answerNotes(two), ['2 quotes in this answer are not in your messages.'])
})

test('an answer that was cut short says so', () => {
  assert.deepEqual(answerNotes({ ...answer, response: { ...response, truncated: true } }), ['This answer was cut short.'])
})

test('an answer that the reader stopped says so, however it was stopped', () => {
  assert.deepEqual(answerNotes({ ...answer, stopped: true }), ['You stopped this answer.'])
  assert.deepEqual(answerNotes({ ...answer, response: { ...response, cancelled: true } }), ['You stopped this answer.'])
})

test('an answer of an older build, with a value that is no list, has no note and does not fail', () => {
  const odd = { ...answer, response: { ...response, unverifiedQuotes: 'not a list' as unknown as string[] } }
  assert.deepEqual(answerNotes(odd), [])
})

test('an answer that is still being written has no note yet', () => {
  assert.deepEqual(answerNotes({ ...answer, streaming: true, stopped: true }), [])
})
