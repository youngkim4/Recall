import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { Job } from '../types.ts'
import { atEnd, jobTimes, logLines, standingOf, stepStates, whatItRead } from './jobFacts.ts'

const job = (fields: Partial<Job>): Job => ({
  id: 'job-1',
  action: 'analyze',
  status: 'running',
  createdAt: '2026-05-01T10:00:00',
  updatedAt: '2026-05-01T10:02:00',
  logs: [],
  ...fields,
})

test('a log is its lines, oldest first, each with the time the server gave', () => {
  const lines = logLines([
    { time: '2026-05-01T10:00:01', message: 'Reading the messages' },
    { time: '2026-05-01T10:00:09', message: 'Writing the report' },
  ])

  assert.deepEqual(lines, [
    { time: '2026-05-01T10:00:01', words: 'Reading the messages' },
    { time: '2026-05-01T10:00:09', words: 'Writing the report' },
  ])
})

test('a line that is only text has no time', () => {
  assert.deepEqual(logLines(['Starting']), [{ time: '', words: 'Starting' }])
})

test("the server's own record of a request is not a line of the job", () => {
  const lines = logLines([
    { time: '2026-05-01T10:00:01', message: '127.0.0.1 - - [01/May/2026 10:00:01] "GET /api/jobs HTTP/1.1" 200 -' },
    '127.0.0.1 - - [01/May/2026 10:00:02] "GET /api/jobs HTTP/1.1" 200 -',
    { time: '2026-05-01T10:00:03', message: 'Reading the messages' },
  ])

  assert.deepEqual(lines.map((line) => line.words), ['Reading the messages'])
})

test('a line with nothing in it is left out', () => {
  assert.deepEqual(logLines([{ time: '2026-05-01T10:00:01', message: '  ' }, '', { time: '2026-05-01T10:00:02' }]), [])
  assert.deepEqual(logLines(undefined), [])
})

test('a running job stands at its step, or at the last thing it wrote', () => {
  const stepped = job({ progress: { step: 2, of: 5, label: 'Reading the messages' }, logs: ['Starting'] })
  const plain = job({ logs: ['Starting', { time: '2026-05-01T10:00:03', message: 'Counting' }] })

  assert.equal(standingOf(stepped), 'Step 2 of 5: Reading the messages')
  assert.equal(standingOf(plain), 'Counting')
})

test('a failed job says why: its error, or the last thing it wrote, or that it did not say', () => {
  assert.equal(standingOf(job({ status: 'failed', error: 'The service did not answer' })), 'The service did not answer')
  assert.equal(standingOf(job({ status: 'failed', logs: ['Lost the connection'] })), 'Lost the connection')
  assert.equal(standingOf(job({ status: 'failed' })), 'It stopped without saying why')
})

test('a job that ended well, or has not begun, has nothing more to say', () => {
  assert.equal(standingOf(job({ status: 'completed', logs: ['Done writing'] })), '')
  assert.equal(standingOf(job({ status: 'queued' })), '')
})

test('the steps before the one in hand are done, and the ones after it are ahead', () => {
  assert.deepEqual(stepStates({ step: 2, of: 5 }), ['done', 'now', 'ahead', 'ahead', 'ahead'])
  assert.deepEqual(stepStates({ step: 5, of: 5 }), ['done', 'done', 'done', 'done', 'now'])
})

test('a step outside its count is held inside it', () => {
  assert.deepEqual(stepStates({ step: 0, of: 3 }), ['now', 'ahead', 'ahead'])
  assert.deepEqual(stepStates({ step: 9, of: 3 }), ['done', 'done', 'now'])
  assert.deepEqual(stepStates({ step: 1, of: 0 }), [])
  // a count no screen could draw is not drawn
  assert.deepEqual(stepStates({ step: 1, of: 500 }), [])
})

test('what an import read is given as figures', () => {
  const figures = whatItRead({ messages: 565666, conversations: 1405, firstYear: 2016, lastYear: 2026, recovered: 1200 })

  assert.deepEqual(
    figures.map((figure) => figure.label),
    ['Messages', 'Conversations', 'Years', 'Restored'],
  )
  assert.equal(figures[2].value, '2016–2026')
  assert.match(figures[0].value, /565.666/)
})

test('a figure the job did not give is not shown', () => {
  assert.deepEqual(whatItRead(null), [])
  assert.deepEqual(whatItRead({ reportPath: '/reports/analysis_x.md' }), [])
  assert.deepEqual(
    whatItRead({ messages: 12, firstYear: 2024, lastYear: 2024, recovered: 0 }).map((figure) => [figure.label, figure.value]),
    [
      ['Messages', '12'],
      ['Years', '2024'],
    ],
  )
  assert.deepEqual(whatItRead({ messages: 3, firstYear: null, lastYear: null }).map((figure) => figure.label), ['Messages'])
})

test('a reader at the end of a log is still at the end within a line of it', () => {
  assert.equal(atEnd({ scrollHeight: 1000, scrollTop: 600, clientHeight: 400 }), true)
  assert.equal(atEnd({ scrollHeight: 1000, scrollTop: 580, clientHeight: 400 }), true)
  assert.equal(atEnd({ scrollHeight: 1000, scrollTop: 300, clientHeight: 400 }), false)
  // a log shorter than its box is all in view
  assert.equal(atEnd({ scrollHeight: 200, scrollTop: 0, clientHeight: 400 }), true)
})

test('a job says when it began, and when it ended or last changed', () => {
  const say = (value: string) => `<${value}>`

  assert.deepEqual(jobTimes(job({ status: 'running' }), say), [
    'Started <2026-05-01T10:00:00>',
    'Last change <2026-05-01T10:02:00>',
  ])
  assert.deepEqual(jobTimes(job({ status: 'completed' }), say), [
    'Started <2026-05-01T10:00:00>',
    'Finished <2026-05-01T10:02:00>',
  ])
  assert.deepEqual(jobTimes(job({ status: 'failed' }), say), [
    'Started <2026-05-01T10:00:00>',
    'Stopped <2026-05-01T10:02:00>',
  ])
})

test('a job that waits has only the time it was asked for', () => {
  assert.deepEqual(jobTimes(job({ status: 'queued' }), (value) => value), ['Asked for 2026-05-01T10:00:00'])
})

test('a time the job does not have is not written', () => {
  assert.deepEqual(jobTimes(job({ status: 'completed', updatedAt: '' }), (value) => value), ['Started 2026-05-01T10:00:00'])
  assert.deepEqual(jobTimes(job({ status: 'completed', createdAt: '', updatedAt: '' }), (value) => value), [])
})
