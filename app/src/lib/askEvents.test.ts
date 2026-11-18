import assert from 'node:assert/strict'
import { test } from 'node:test'
import { readFrame, takeFrames } from './askEvents.ts'

const frame = (event: unknown) => `data: ${JSON.stringify(event)}`

test('whole frames are taken from what has arrived, and the rest waits', () => {
  assert.deepEqual(takeFrames('data: {"a":1}\n\ndata: {"b":2}\n\ndata: {"c"'), {
    frames: ['data: {"a":1}', 'data: {"b":2}'],
    rest: 'data: {"c"',
  })
  assert.deepEqual(takeFrames('data: {"a"'), { frames: [], rest: 'data: {"a"' })
  assert.deepEqual(takeFrames(''), { frames: [], rest: '' })
  assert.deepEqual(takeFrames('data: {"a":1}\n\n'), { frames: ['data: {"a":1}'], rest: '' })
})

test('a status says what the server is doing', () => {
  assert.deepEqual(readFrame(frame({ type: 'status', text: 'Reading messages' })), {
    type: 'status',
    text: 'Reading messages',
    reset: false,
  })
})

test('a part of a conversation being read is progress', () => {
  assert.deepEqual(
    readFrame(frame({ type: 'progress', tool: 'conversation_digest', conversation: 'A', part: 2, of: 4, label: 'May', state: 'done' })),
    { type: 'progress', part: 2, of: 4, state: 'done' },
  )
  // progress with no numbers carries nothing to show
  assert.equal(readFrame(frame({ type: 'progress', part: 'two', of: 4, state: 'done' })), null)
})

test('a status can say that the text so far was not the answer', () => {
  assert.deepEqual(readFrame(frame({ type: 'status', text: 'Searching again', reset: true })), {
    type: 'status',
    text: 'Searching again',
    reset: true,
  })
  // with no words it still clears the text
  assert.deepEqual(readFrame(frame({ type: 'status', reset: true })), { type: 'status', text: '', reset: true })
  // only true clears
  assert.deepEqual(readFrame(frame({ type: 'status', text: 'a', reset: 1 })), { type: 'status', text: 'a', reset: false })
})

test('a status with nothing in it is no event', () => {
  assert.equal(readFrame(frame({ type: 'status' })), null)
  assert.equal(readFrame(frame({ type: 'status', text: '' })), null)
})

test('a delta carries the next piece of the answer', () => {
  assert.deepEqual(readFrame(frame({ type: 'delta', text: ' and then' })), { type: 'delta', text: ' and then' })
  assert.equal(readFrame(frame({ type: 'delta', text: '' })), null)
  assert.equal(readFrame(frame({ type: 'delta' })), null)
})

test('the end carries the whole answer', () => {
  const payload = { question: 'q', terms: [], answer: 'a', citations: [], cancelled: true }

  assert.deepEqual(readFrame(frame({ type: 'done', payload })), { type: 'done', payload })
  assert.equal(readFrame(frame({ type: 'done' })), null)
  assert.equal(readFrame(frame({ type: 'done', payload: 'text' })), null)
})

test('an error carries the sentence, the fixed word and the seconds to wait', () => {
  const event = { type: 'error', code: 'rate_limit', error: 'Too many questions. Wait a moment.', retry_after: 12 }

  assert.deepEqual(readFrame(frame(event)), {
    type: 'error',
    code: 'rate_limit',
    error: 'Too many questions. Wait a moment.',
    retryAfter: 12,
    body: event,
  })
})

test('an error that says nothing still has a sentence', () => {
  const read = readFrame(frame({ type: 'error' }))

  assert.equal(read?.type, 'error')
  assert.deepEqual(read, { type: 'error', code: '', error: 'The question could not be answered.', retryAfter: undefined, body: { type: 'error' } })
})

test('a frame that cannot be read is no event', () => {
  assert.equal(readFrame('data: {not json'), null)
  assert.equal(readFrame(': keep-alive'), null)
  assert.equal(readFrame(''), null)
  assert.equal(readFrame('data: 7'), null)
  assert.equal(readFrame('data: null'), null)
  assert.equal(readFrame(frame({ type: 'something_new', text: 'x' })), null)
})

test('the data of a frame is found among its other lines', () => {
  assert.deepEqual(readFrame(`event: message\n${frame({ type: 'delta', text: 'x' })}`), { type: 'delta', text: 'x' })
  assert.deepEqual(readFrame('data:{"type":"delta","text":"x"}'), { type: 'delta', text: 'x' })
})
