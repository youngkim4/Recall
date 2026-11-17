import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { ModelChoice } from '../types.ts'
import { offerModels, qualityOptions } from './models.ts'

// invented models: the app knows no model by its id
const LUNA: ModelChoice = { id: 'model-luna', label: 'Model Luna', level: 'low' }
const TERRA: ModelChoice = { id: 'model-terra', label: 'Model Terra', level: 'medium' }
const SOL: ModelChoice = { id: 'model-sol', label: 'Model Sol', level: 'high' }
const CHOICES = [LUNA, TERRA, SOL]

test('the models a server offers are quality levels, in the order the server gives', () => {
  assert.deepEqual(qualityOptions(['model-sol', 'model-luna', 'model-terra'], CHOICES), [
    { value: 'model-luna', label: 'Low', name: 'Model Luna', note: 'Fastest, lowest cost' },
    { value: 'model-terra', label: 'Medium', name: 'Model Terra', note: 'A balance of speed and care' },
    { value: 'model-sol', label: 'High', name: 'Model Sol', note: 'Most careful, highest cost' },
  ])
})

test('only the models asked for are offered', () => {
  assert.deepEqual(
    qualityOptions(['model-sol'], CHOICES).map((option) => option.label),
    ['High'],
  )
  assert.deepEqual(qualityOptions([], CHOICES), [])
})

test('a level above High is offered as Top', () => {
  const nova: ModelChoice = { id: 'model-nova', label: 'Model Nova', level: 'top' }

  assert.deepEqual(qualityOptions(['model-nova', 'model-sol'], [...CHOICES, nova]).slice(1), [
    { value: 'model-nova', label: 'Top', name: 'Model Nova', note: '' },
  ])
})

test('a model the server does not describe goes after the levels, under its own id', () => {
  assert.deepEqual(qualityOptions(['model-other', 'model-terra'], CHOICES), [
    { value: 'model-terra', label: 'Medium', name: 'Model Terra', note: 'A balance of speed and care' },
    { value: 'model-other', label: 'model-other', name: '', note: '' },
  ])
})

test('a level this build does not know is offered under the name of its model', () => {
  const odd = { id: 'model-odd', label: 'Model Odd', level: 'utmost' } as unknown as ModelChoice

  assert.deepEqual(qualityOptions(['model-odd'], [odd]), [
    { value: 'model-odd', label: 'Model Odd', name: '', note: '' },
  ])
})

test('a list of choices that is not a list offers each model under its own id', () => {
  const none = { broken: true } as unknown as ModelChoice[]

  assert.deepEqual(qualityOptions(['model-sol'], none), [
    { value: 'model-sol', label: 'model-sol', name: '', note: '' },
  ])
})

test('the choices of the server are kept for a caller that has only the ids', () => {
  offerModels(CHOICES)
  assert.deepEqual(
    qualityOptions(['model-luna', 'model-sol']).map((option) => `${option.label} ${option.name}`),
    ['Low Model Luna', 'High Model Sol'],
  )

  // the list that was given is not the list that is kept
  const given = [SOL]
  offerModels(given)
  given.pop()
  assert.deepEqual(
    qualityOptions(['model-sol']).map((option) => option.label),
    ['High'],
  )

  // a server that sends no choices leaves the ids
  offerModels(undefined)
  assert.deepEqual(
    qualityOptions(['model-sol']).map((option) => option.label),
    ['model-sol'],
  )
})
