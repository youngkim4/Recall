import assert from 'node:assert/strict'
import { test } from 'node:test'
import { ApiError, apiErrorFrom, errorCode, errorSentence } from './apiError.ts'

test('a refusal carries the sentence, the fixed word, the status and the whole answer', () => {
  const body = { error: 'Recall is getting your archive ready. This takes a moment.', code: 'archive_building', job: { id: 'job-7' } }

  const error = apiErrorFrom(503, 'Service Unavailable', body)

  assert.equal(error instanceof ApiError, true)
  assert.equal(error instanceof Error, true)
  assert.equal(error.name, 'ApiError')
  assert.equal(error.message, 'Recall is getting your archive ready. This takes a moment.')
  assert.equal(error.code, 'archive_building')
  assert.equal(error.status, 503)
  assert.deepEqual(error.body, body)
})

test('a refusal with no fixed word has none', () => {
  const error = apiErrorFrom(409, 'Conflict', { error: 'An import is already running.' })

  assert.equal(error.message, 'An import is already running.')
  assert.equal(error.code, undefined)
  assert.equal(errorCode(error), '')
})

test('an answer that says nothing leaves the status as the sentence', () => {
  assert.equal(apiErrorFrom(502, 'Bad Gateway', null).message, '502 Bad Gateway')
  assert.equal(apiErrorFrom(500, '', 'not json').message, '500')
  assert.equal(apiErrorFrom(500, 'Internal Server Error', { error: '' }).message, '500 Internal Server Error')
  assert.equal(apiErrorFrom(500, 'Internal Server Error', { error: 7, code: 9 }).code, undefined)
  assert.equal(apiErrorFrom(400, 'Bad Request', ['a list']).message, '400 Bad Request')
})

test('the seconds to wait are read when the server gives a number', () => {
  assert.equal(apiErrorFrom(429, 'Too Many Requests', { error: 'Wait.', code: 'rate_limit', retry_after: 20 }).retryAfter, 20)
  assert.equal(apiErrorFrom(429, 'Too Many Requests', { error: 'Wait.', retry_after: 'soon' }).retryAfter, undefined)
  assert.equal(apiErrorFrom(429, 'Too Many Requests', { error: 'Wait.', retry_after: -1 }).retryAfter, undefined)
})

test('the fixed word of an error is read from any thing that was thrown', () => {
  assert.equal(errorCode(new ApiError('No key.', { status: 0, code: 'no_key' })), 'no_key')
  assert.equal(errorCode(new Error('plain')), '')
  assert.equal(errorCode('text'), '')
  assert.equal(errorCode(null), '')
})

test('the sentence of the server is shown as it is', () => {
  assert.equal(errorSentence(new ApiError('The key was refused.', { status: 0, code: 'key_rejected' }), 'Fallback.'), 'The key was refused.')
  assert.equal(errorSentence(new Error('The answer stream ended.'), 'Fallback.'), 'The answer stream ended.')
})

test('a request that reached no server has no sentence of the server to show', () => {
  // the browser throws a TypeError with words of its own
  assert.equal(errorSentence(new TypeError('Failed to fetch'), 'Recall did not answer.'), 'Recall did not answer.')
  assert.equal(errorSentence('text', 'Recall did not answer.'), 'Recall did not answer.')
  assert.equal(errorSentence(new Error(''), 'Recall did not answer.'), 'Recall did not answer.')
})
