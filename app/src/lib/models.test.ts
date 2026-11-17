import assert from 'node:assert/strict'
import { test } from 'node:test'
import { qualityOptions } from './models.ts'

test('the models a server offers are quality levels, lowest first', () => {
  assert.deepEqual(qualityOptions(['gpt-5.5', 'gpt-5.4-mini', 'gpt-5.4']), [
    { value: 'gpt-5.4-mini', label: 'Low', name: 'GPT-5.4 mini', note: 'Fastest, lowest cost' },
    { value: 'gpt-5.4', label: 'Medium', name: 'GPT-5.4', note: 'A balance of speed and care' },
    { value: 'gpt-5.5', label: 'High', name: 'GPT-5.5', note: 'Most careful, highest cost' },
  ])
})

test('only the levels the server can run are offered', () => {
  assert.deepEqual(
    qualityOptions(['gpt-5.5']).map((option) => option.label),
    ['High'],
  )
  assert.deepEqual(qualityOptions([]), [])
})

test('a model with no level goes after the levels, under its own name', () => {
  const options = qualityOptions(['gpt-4o-mini', 'gpt-5.4'])

  assert.deepEqual(options, [
    { value: 'gpt-5.4', label: 'Medium', name: 'GPT-5.4', note: 'A balance of speed and care' },
    { value: 'gpt-4o-mini', label: 'GPT-4o mini', name: '', note: '' },
  ])
})

const nameOf = (model: string) => qualityOptions([model])[0].label

test('a numbered GPT model is written the way its maker writes it', () => {
  assert.equal(nameOf('gpt-4.1'), 'GPT-4.1')
  assert.equal(nameOf('gpt-4.1-nano'), 'GPT-4.1 nano')
  assert.equal(nameOf('GPT-4o'), 'GPT-4o')
  assert.equal(nameOf('gpt-4-turbo-preview'), 'GPT-4 turbo preview')
})

test('any other model keeps the name it came with', () => {
  assert.equal(nameOf('gpt-oss-120b'), 'gpt-oss-120b')
  assert.equal(nameOf('o3-mini'), 'o3-mini')
  assert.equal(nameOf('gptx'), 'gptx')
})
