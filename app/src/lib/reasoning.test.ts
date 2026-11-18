import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { ModelChoice } from '../types.ts'
import { fakeStorage, noPage, pageWithNoStorage, pageWithStorage } from './fakeStorage.test.ts'
import { effortOf, readEfforts, reasoningSteps, storeEffort, subscribeEfforts } from './reasoning.ts'

const KEY = 'recall.reasoning.v1'

// invented models: the app knows no model by its id
const LUNA: ModelChoice = {
  id: 'model-luna',
  label: 'Model Luna',
  level: 'low',
  reasoningLevels: ['none', 'low', 'medium', 'high', 'xhigh', 'max'],
  reasoningDefault: 'medium',
}
const SOL: ModelChoice = {
  id: 'model-sol',
  label: 'Model Sol',
  level: 'high',
  reasoningLevels: ['low', 'medium', 'high', 'xhigh', 'max'],
  reasoningDefault: 'medium',
}
const PLAIN: ModelChoice = { id: 'model-plain', label: 'Model Plain', level: 'medium' }

test('the levels of a model are steps, least first, each with a name a person reads', () => {
  assert.deepEqual(reasoningSteps(LUNA), [
    { value: 'none', label: 'Off' },
    { value: 'low', label: 'Low' },
    { value: 'medium', label: 'Medium' },
    { value: 'high', label: 'High' },
    { value: 'xhigh', label: 'Extra high' },
    { value: 'max', label: 'Max' },
  ])
  assert.deepEqual(
    reasoningSteps(SOL).map((step) => step.value),
    ['low', 'medium', 'high', 'xhigh', 'max'],
  )
})

test('a level this build has no name for is shown by its own word', () => {
  assert.deepEqual(reasoningSteps({ ...SOL, reasoningLevels: ['low', 'ultra_deep'] })[1], {
    value: 'ultra_deep',
    label: 'Ultra deep',
  })
})

test('a model with no levels, or none that the server names, has no steps', () => {
  assert.deepEqual(reasoningSteps(PLAIN), [])
  assert.deepEqual(reasoningSteps(undefined), [])
  assert.deepEqual(reasoningSteps({ ...SOL, reasoningLevels: 'low' as unknown as string[] }), [])
  assert.deepEqual(reasoningSteps({ ...SOL, reasoningLevels: ['low', 7 as unknown as string, ''] }), [{ value: 'low', label: 'Low' }])
})

test('the effort of a model is the one kept for it, when the model still has that level', () => {
  assert.equal(effortOf(LUNA, { 'model-luna': 'none' }), 'none')
  assert.equal(effortOf(SOL, { 'model-luna': 'none', 'model-sol': 'max' }), 'max')
  // each model keeps its own
  assert.equal(effortOf(SOL, { 'model-luna': 'none' }), 'medium')
})

test('a kept effort that the model no longer has gives way to the default of the model', () => {
  assert.equal(effortOf(SOL, { 'model-sol': 'none' }), 'medium')
  assert.equal(effortOf(SOL, {}), 'medium')
})

test('with no default that the model lists, the level in the middle is asked for', () => {
  // what a control shows is what is sent
  assert.equal(effortOf({ ...SOL, reasoningDefault: null }, {}), 'high')
  assert.equal(effortOf({ ...SOL, reasoningDefault: 'none' }, {}), 'high')
  assert.equal(effortOf({ ...LUNA, reasoningDefault: undefined }, {}), 'medium')
  assert.equal(effortOf({ ...SOL, reasoningLevels: ['low', 'high'], reasoningDefault: null }, {}), 'low')
})

test('a model with no levels is asked for nothing', () => {
  assert.equal(effortOf(PLAIN, { 'model-plain': 'high' }), '')
  assert.equal(effortOf(undefined, {}), '')
})

test('an effort is kept for its model and read back', () => {
  const storage = fakeStorage()
  pageWithStorage(storage)

  storeEffort('model-luna', 'none')
  storeEffort('model-sol', 'max')

  assert.deepEqual(readEfforts(), { 'model-luna': 'none', 'model-sol': 'max' })
  assert.deepEqual(JSON.parse(storage.getItem(KEY) ?? ''), { 'model-luna': 'none', 'model-sol': 'max' })
})

test('keeping an effort makes a new record and tells whoever listens', () => {
  pageWithStorage(fakeStorage())
  const before = readEfforts()
  let told = 0
  const stop = subscribeEfforts(() => {
    told += 1
  })

  storeEffort('model-sol', 'high')
  const after = readEfforts()
  // the same effort again changes nothing
  storeEffort('model-sol', 'high')

  assert.notEqual(after, before)
  assert.equal(readEfforts(), after)
  assert.equal(told, 1)
  stop()
  storeEffort('model-sol', 'low')
  assert.equal(told, 1)
})

test('what was kept is read only when it is a record of short plain words', () => {
  for (const kept of ['not json', '[]', '"low"', 'null', '{"model-sol": 5}', '{"model-sol": "LOW!"}', `{"model-sol": "${'x'.repeat(40)}"}`]) {
    pageWithStorage(fakeStorage({ [KEY]: kept }))
    assert.deepEqual(readEfforts(), {}, kept)
  }
  pageWithStorage(fakeStorage({ [KEY]: '{"model-sol": "xhigh", "model-luna": 7}' }))
  assert.deepEqual(readEfforts(), { 'model-sol': 'xhigh' })
})

test('a word that is no level is not kept', () => {
  const storage = fakeStorage()
  pageWithStorage(storage)

  storeEffort('model-sol', 'Very High!')
  storeEffort('', 'low')

  assert.equal(storage.getItem(KEY), null)
})

test('with no page or no storage a level still holds while the page is open', () => {
  noPage()
  assert.deepEqual(readEfforts(), {})
  storeEffort('model-sol', 'low')
  assert.deepEqual(readEfforts(), { 'model-sol': 'low' })

  pageWithNoStorage()
  storeEffort('model-sol', 'max')
  assert.deepEqual(readEfforts(), { 'model-sol': 'max' })
})

test('a store that is full does not stop the control: the level holds and whoever listens is told', () => {
  const full = fakeStorage({ [KEY]: '{"model-luna":"none"}' }, { roomFor: 0 })
  pageWithStorage(full)
  let told = 0
  const stop = subscribeEfforts(() => {
    told += 1
  })

  storeEffort('model-sol', 'xhigh')

  assert.deepEqual(readEfforts(), { 'model-luna': 'none', 'model-sol': 'xhigh' })
  assert.equal(readEfforts(), readEfforts())
  assert.equal(told, 1)
  assert.equal(full.getItem(KEY), '{"model-luna":"none"}')
  stop()
})

test('levels that one store refused do not show on another store', () => {
  pageWithStorage(fakeStorage({}, { roomFor: 0 }))
  storeEffort('model-sol', 'xhigh')

  pageWithStorage(fakeStorage({ [KEY]: '{"model-sol":"low"}' }))

  assert.deepEqual(readEfforts(), { 'model-sol': 'low' })
})
