import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { FirstRunStatus, FirstRunStep, Job } from '../types.ts'
import { chatPhase, costCeiling, keyLine, progressLine, stepProblems } from './firstRun.ts'

const step = (name: FirstRunStep['name'], change: Partial<FirstRunStep> = {}): FirstRunStep => ({
  name,
  state: 'none',
  estimatedCost: 1,
  error: null,
  code: null,
  ...change,
})

const status = (change: Partial<FirstRunStatus> = {}): FirstRunStatus => ({
  key: { present: true, source: 'file' },
  archive: { holdsMessages: true, conversations: 10, messages: 500 },
  steps: [step('meaning'), step('memory'), step('people')],
  estimatedCost: 3,
  ready: false,
  jobId: null,
  ...change,
})

const job = (change: Partial<Job> = {}): Job => ({
  id: 'j1',
  action: 'first_run',
  status: 'running',
  createdAt: '',
  updatedAt: '',
  logs: [],
  ...change,
})

test('an upper bound is rounded up to the cent, never down', () => {
  assert.equal(costCeiling(6.701), 6.71)
  assert.equal(costCeiling(6.71), 6.71)
  assert.equal(costCeiling(0), 0)
})

test('what chat is ready for follows the job first, then the status', () => {
  assert.equal(chatPhase(null, null), 'unknown')
  assert.equal(chatPhase(status(), null), 'not_ready')
  assert.equal(chatPhase(status({ ready: true }), null), 'ready')
  assert.equal(
    chatPhase(status({ archive: { holdsMessages: false, conversations: 0, messages: 0 } }), null),
    'no_messages',
  )
  assert.equal(chatPhase(status(), job()), 'running')
  assert.equal(chatPhase(status(), job({ status: 'queued' })), 'running')
  assert.equal(chatPhase(status(), job({ status: 'failed', error: 'No key.' })), 'failed')
  // a job that ended well leaves the status, asked again, to say what is ready
  assert.equal(chatPhase(status({ ready: true }), job({ status: 'completed' })), 'ready')
})

test('a step whose cost is not known says why, and a ready step says nothing', () => {
  const found = status({
    steps: [
      step('meaning', { state: 'ready', error: 'ignored' }),
      step('memory', { error: 'The notes file could not be read.', estimatedCost: 0 }),
      step('people'),
    ],
  })
  assert.deepEqual(stepProblems(found), ['The notes file could not be read.'])
})

test('the progress line names the step and how far it is', () => {
  assert.equal(
    progressLine(job({ progress: { step: 2, of: 3, label: 'Reading your conversations', done: 40, total: 2300 } })),
    'Reading your conversations: 40 of 2,300',
  )
  assert.equal(progressLine(job({ progress: { step: 1, of: 3, label: 'Starting' } })), 'Starting')
  assert.equal(progressLine(job()), '')
})

test('the key is told by where it is, never by what it is', () => {
  assert.equal(keyLine({ present: true, source: 'file' }), 'Saved')
  assert.equal(keyLine({ present: true, source: 'environment' }), 'Set where Recall starts')
  assert.equal(keyLine({ present: false, source: null }), 'None')
})
