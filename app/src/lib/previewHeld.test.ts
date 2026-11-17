import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { PreviewPayload } from '../types.ts'
import { NOTHING_HELD, shownPreview, type Held } from './previewHeld.ts'

const figures = { stats: { totalMessages: 7 } } as unknown as PreviewPayload

const held = (fields: Partial<Held> = {}): Held => ({
  key: 'archive::sam::model::::',
  subject: 'archive\nsam',
  generation: 3,
  payload: figures,
  ...fields,
})

const asked = (fields: Partial<Omit<Held, 'payload'>> = {}) => ({
  key: 'archive::sam::model::::',
  subject: 'archive\nsam',
  generation: 3,
  ...fields,
})

test('the figures that were asked for are shown and settled', () => {
  assert.deepEqual(shownPreview(held(), asked()), { payload: figures, settled: true })
})

test('other dates of the same conversation keep the figures up until theirs arrive', () => {
  const otherDates = asked({ key: 'archive::sam::model::2024-01-01::' })

  assert.deepEqual(shownPreview(held(), otherDates), { payload: figures, settled: false })
})

test('the figures of another conversation are not shown', () => {
  const other = asked({ key: 'archive::avery::model::::', subject: 'archive\navery' })

  assert.deepEqual(shownPreview(held(), other), { payload: null, settled: false })
})

test('after the saved figures were cleared, the figures in memory are not shown', () => {
  // an import wrote another archive under the same path: same key, same subject
  assert.deepEqual(shownPreview(held(), asked({ generation: 4 })), { payload: null, settled: false })
})

test('figures that arrive after the clearing are shown', () => {
  assert.deepEqual(shownPreview(held({ generation: 4 }), asked({ generation: 4 })), { payload: figures, settled: true })
})

test('with nothing asked for, or nothing held, nothing is shown', () => {
  assert.deepEqual(shownPreview(held(), asked({ key: '', subject: '' })), { payload: null, settled: false })
  assert.deepEqual(shownPreview(NOTHING_HELD, asked()), { payload: null, settled: false })
  assert.deepEqual(shownPreview(NOTHING_HELD, asked({ key: '', subject: '', generation: 0 })), { payload: null, settled: false })
  assert.deepEqual(shownPreview(held({ payload: null }), asked()), { payload: null, settled: false })
})
