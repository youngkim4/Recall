import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { PreviewPayload } from '../types'
import {
  clearPreviewCache,
  isPreviewPayload,
  makePreviewCacheKey,
  previewCacheStats,
  readPreviewCache,
  writePreviewCache,
} from './cache.ts'
import { fakeStorage, noPage, pageWithNoStorage, pageWithStorage } from './fakeStorage.test.ts'

const STORE_KEY = 'recall.previewCache.v7'
const DAY = 86_400_000
const T0 = Date.UTC(2026, 10, 1)

// the least a preview must have to be read; `total` tells one from another
const preview = (total = 1) =>
  ({
    stats: { totalMessages: total },
    activity: { unit: 'month', points: [] },
    daily: { start: '', counts: [] },
    recentMessages: [],
  }) as unknown as PreviewPayload

const totalOf = (payload: PreviewPayload | null) => payload?.stats.totalMessages

test('a preview has the parts this build reads', () => {
  assert.equal(isPreviewPayload(preview()), true)
  assert.equal(isPreviewPayload(null), false)
  assert.equal(isPreviewPayload('preview'), false)
  assert.equal(isPreviewPayload({}), false)
  assert.equal(isPreviewPayload({ ...preview(), activity: undefined }), false)
  assert.equal(isPreviewPayload({ ...preview(), activity: { unit: 'month' } }), false)
  assert.equal(isPreviewPayload({ ...preview(), daily: { start: '' } }), false)
  assert.equal(isPreviewPayload({ ...preview(), recentMessages: undefined }), false)
  assert.equal(isPreviewPayload({ ...preview(), stats: null }), false)
})

test('a key names the archive, the conversation, the quality and the dates', () => {
  const base = { messagesPath: '/archive/messages.csv', contact: '+15550100199', model: 'gpt-5.4' }

  assert.equal(makePreviewCacheKey(base), '/archive/messages.csv::+15550100199::gpt-5.4::::')
  assert.notEqual(makePreviewCacheKey({ ...base, since: '2024-01-01' }), makePreviewCacheKey(base))
  assert.notEqual(makePreviewCacheKey({ ...base, until: '2024-01-01' }), makePreviewCacheKey({ ...base, since: '2024-01-01' }))
})

test('a preview that was kept is read back under its key', () => {
  pageWithStorage(fakeStorage())

  writePreviewCache('a', preview(7), T0)

  assert.equal(totalOf(readPreviewCache('a', T0)), 7)
  assert.equal(readPreviewCache('b', T0), null)
})

test('nothing is kept or read under an empty key', () => {
  const storage = pageWithStorage(fakeStorage())

  writePreviewCache('', preview(), T0)

  assert.equal(storage.changes(), 0)
  assert.equal(readPreviewCache('', T0), null)
})

test('a preview is of no use after two weeks', () => {
  pageWithStorage(fakeStorage())
  writePreviewCache('a', preview(), T0)

  assert.notEqual(readPreviewCache('a', T0 + 14 * DAY), null)
  assert.equal(readPreviewCache('a', T0 + 14 * DAY + 1), null)
  assert.deepEqual(previewCacheStats(T0 + 14 * DAY + 1), { count: 0, lastSavedAt: '' })
})

test('the newest previews are kept and the oldest give way', () => {
  pageWithStorage(fakeStorage())

  Array.from({ length: 97 }, (_, index) => index).forEach((index) => {
    writePreviewCache(`key-${index}`, preview(index), T0 + index)
  })

  assert.equal(readPreviewCache('key-0', T0 + 100), null)
  assert.equal(totalOf(readPreviewCache('key-1', T0 + 100)), 1)
  assert.equal(totalOf(readPreviewCache('key-96', T0 + 100)), 96)
  assert.deepEqual(previewCacheStats(T0 + 100), { count: 96, lastSavedAt: new Date(T0 + 96).toISOString() })
})

test('keeping a preview again replaces it and counts as its newest', () => {
  pageWithStorage(fakeStorage())
  writePreviewCache('a', preview(1), T0)
  writePreviewCache('b', preview(2), T0 + 1)

  writePreviewCache('a', preview(3), T0 + 2)

  assert.equal(totalOf(readPreviewCache('a', T0 + 3)), 3)
  assert.deepEqual(previewCacheStats(T0 + 3), { count: 2, lastSavedAt: new Date(T0 + 2).toISOString() })
})

test('reading and counting never write to the store', () => {
  const storage = pageWithStorage(fakeStorage())
  writePreviewCache('a', preview(), T0)
  const before = storage.changes()

  readPreviewCache('a', T0 + 1)
  readPreviewCache('missing', T0 + 1)
  readPreviewCache('a', T0 + 30 * DAY)
  previewCacheStats(T0 + 1)

  assert.equal(storage.changes(), before)
})

test('a damaged entry is passed over and the whole ones are still read', () => {
  const whole = { payload: preview(5), savedAt: T0 }
  const storage = pageWithStorage(
    fakeStorage({
      [STORE_KEY]: JSON.stringify({
        empty: null,
        words: 'not an entry',
        timeless: { payload: preview(), savedAt: 'yesterday' },
        endless: { payload: preview(), savedAt: 1e300 },
        shapeless: { payload: { stats: {} }, savedAt: T0 },
        whole,
      }),
    }),
  )

  assert.equal(readPreviewCache('empty', T0), null)
  assert.equal(readPreviewCache('timeless', T0), null)
  assert.equal(readPreviewCache('endless', T0), null)
  assert.equal(readPreviewCache('shapeless', T0), null)
  assert.equal(totalOf(readPreviewCache('whole', T0)), 5)
  assert.deepEqual(previewCacheStats(T0), { count: 1, lastSavedAt: new Date(T0).toISOString() })

  // the next write leaves the damaged ones behind
  writePreviewCache('new', preview(6), T0 + 1)
  assert.deepEqual(Object.keys(JSON.parse(storage.dump()[STORE_KEY])).sort(), ['new', 'whole'])
})

test('a store that is not a store of previews reads as empty', () => {
  ;['{{{', '[1, 2]', '7', 'null', '"text"'].forEach((raw) => {
    pageWithStorage(fakeStorage({ [STORE_KEY]: raw }))

    assert.equal(readPreviewCache('a', T0), null, raw)
    assert.deepEqual(previewCacheStats(T0), { count: 0, lastSavedAt: '' }, raw)
    writePreviewCache('a', preview(2), T0)
    assert.equal(totalOf(readPreviewCache('a', T0)), 2, raw)
  })
})

test('previews kept in an older shape give their space back', () => {
  const storage = pageWithStorage(
    fakeStorage({
      'recall.previewCache.v5': '{}',
      'recall.previewCache.v6': '{"a":{}}',
      'recall.theme': 'dark',
    }),
  )

  writePreviewCache('a', preview(), T0)

  assert.deepEqual(Object.keys(storage.dump()).sort(), ['recall.previewCache.v7', 'recall.theme'])
})

test('clearing drops every preview and nothing else', () => {
  const storage = pageWithStorage(fakeStorage({ 'recall.theme': 'dark' }))
  writePreviewCache('a', preview(), T0)

  clearPreviewCache()

  assert.equal(readPreviewCache('a', T0), null)
  assert.deepEqual(storage.dump(), { 'recall.theme': 'dark' })
})

test('a full store keeps the newest previews that fit', () => {
  const one = JSON.stringify({ k0: { payload: preview(0), savedAt: T0 } }).length
  // room for three previews and not for four
  const storage = pageWithStorage(fakeStorage({}, { roomFor: one * 3 + 8 }))

  Array.from({ length: 6 }, (_, index) => index).forEach((index) => {
    writePreviewCache(`k${index}`, preview(index), T0 + index)
  })

  const held = Object.keys(JSON.parse(storage.dump()[STORE_KEY]))
  assert.ok(held.includes('k5'), `the newest is kept: ${held.join(' ')}`)
  assert.ok(held.length >= 1 && held.length <= 3, `only what fits is kept: ${held.join(' ')}`)
  assert.equal(held.includes('k0'), false)
  assert.equal(totalOf(readPreviewCache('k5', T0 + 10)), 5)
})

test('a page that may not keep anything works as if nothing was kept', () => {
  pageWithNoStorage()

  assert.doesNotThrow(() => writePreviewCache('a', preview(), T0))
  assert.equal(readPreviewCache('a', T0), null)
  assert.deepEqual(previewCacheStats(T0), { count: 0, lastSavedAt: '' })
  assert.doesNotThrow(() => clearPreviewCache())
})

test('a store that fails on every call works as if nothing was kept', () => {
  pageWithStorage(fakeStorage({}, { broken: true }))

  assert.doesNotThrow(() => writePreviewCache('a', preview(), T0))
  assert.equal(readPreviewCache('a', T0), null)
  assert.deepEqual(previewCacheStats(T0), { count: 0, lastSavedAt: '' })
  assert.doesNotThrow(() => clearPreviewCache())
})

test('outside a browser nothing is kept and nothing fails', () => {
  noPage()

  assert.doesNotThrow(() => writePreviewCache('a', preview(), T0))
  assert.equal(readPreviewCache('a', T0), null)
  assert.deepEqual(previewCacheStats(T0), { count: 0, lastSavedAt: '' })
  assert.doesNotThrow(() => clearPreviewCache())
})
