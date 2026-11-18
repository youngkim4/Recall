import assert from 'node:assert/strict'
import { test } from 'node:test'
import { startWork, withProgress, withStatus, workSummary, workTime, type Work } from './chatWork.ts'

const T0 = 1_000_000

test('each new status is a step, and the one before it is done', () => {
  const work = withStatus(withStatus(startWork(T0), 'Reading your archive'), 'Searching for “lake”')
  assert.deepEqual(
    work.steps.map((step) => [step.text, step.state]),
    [
      ['Reading your archive', 'done'],
      ['Searching for “lake”', 'active'],
    ],
  )
})

test('the same status twice in a row is one step', () => {
  const work = withStatus(withStatus(startWork(T0), 'Thinking…'), 'Thinking…')
  assert.equal(work.steps.length, 1)
})

test('the parts of a reading count on the step that is reading', () => {
  let work: Work = withStatus(startWork(T0), 'Reading your whole conversation')
  work = withProgress(work, { part: 1, of: 3, state: 'start' })
  assert.deepEqual([work.steps[0].read, work.steps[0].of], [0, 3])
  work = withProgress(work, { part: 1, of: 3, state: 'done' })
  work = withProgress(work, { part: 2, of: 3, state: 'cached' })
  assert.deepEqual([work.steps[0].read, work.steps[0].of, work.steps[0].state], [2, 3, 'active'])
  work = withProgress(work, { part: 3, of: 3, state: 'failed' })
  // a part that failed is counted as passed, so the bar can end
  assert.deepEqual([work.steps[0].read, work.steps[0].state], [3, 'active'])
})

test('progress with no step yet makes one, and bad numbers change nothing', () => {
  const work = withProgress(startWork(T0), { part: 1, of: 2, state: 'done' })
  assert.equal(work.steps.length, 1)
  assert.equal(work.steps[0].read, 1)
  const same = withProgress(work, { part: Number.NaN, of: -1, state: 'done' })
  assert.deepEqual(same, work)
})

test('the summary says how long the work took, in whole seconds, and lists its steps', () => {
  const work = withStatus(withStatus(startWork(T0), 'Reading your archive'), 'Searching for “lake”')
  assert.deepEqual(workSummary(work, T0 + 6_400), { seconds: 6, steps: ['Reading your archive', 'Searching for “lake”'] })
  // never less than one second
  assert.equal(workSummary(work, T0 + 200)?.seconds, 1)
})

test('work with no steps has no summary', () => {
  assert.equal(workSummary(startWork(T0), T0 + 5_000), undefined)
})

test('how long the work took reads in seconds, then minutes and seconds', () => {
  assert.equal(workTime(6), '6s')
  assert.equal(workTime(59), '59s')
  assert.equal(workTime(80), '1m 20s')
  assert.equal(workTime(120), '2m')
})
