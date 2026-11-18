import assert from 'node:assert/strict'
import { test } from 'node:test'
import { NO_FILTERS, filterParams, filtersKey, filtersSet, shownWord, whereChoices, whereOf, withWhere } from './searchFilters.ts'

test('one picker says where to search: everywhere, a kind of chat, or one conversation', () => {
  const choices = whereChoices([
    { value: '', label: 'All conversations' },
    { value: 'chat-1', label: 'One', meta: '12' },
  ])
  assert.deepEqual(
    choices.map((choice) => choice.value),
    ['', 'kind:direct', 'kind:group', 'chat-1'],
  )
  assert.equal(whereOf(NO_FILTERS), '')
  assert.equal(whereOf({ ...NO_FILTERS, kind: 'group' }), 'kind:group')
  assert.equal(whereOf({ ...NO_FILTERS, contact: 'chat-1' }), 'chat-1')
})

test('choosing where sets one of conversation and kind, and clears the other', () => {
  const inGroup = withWhere({ ...NO_FILTERS, contact: 'chat-1', from: 'me' }, 'kind:group')
  assert.deepEqual(inGroup, { ...NO_FILTERS, kind: 'group', from: 'me' })
  assert.deepEqual(withWhere(inGroup, 'chat-1'), { ...NO_FILTERS, contact: 'chat-1', from: 'me' })
  assert.deepEqual(withWhere(inGroup, ''), { ...NO_FILTERS, from: 'me' })
})

test('no filter is set at first, and the order is not a filter', () => {
  assert.equal(filtersSet(NO_FILTERS), false)
  assert.equal(filtersSet({ ...NO_FILTERS, order: 'oldest' }), false)
  assert.equal(filtersSet({ ...NO_FILTERS, from: 'me' }), true)
  assert.equal(filtersSet({ ...NO_FILTERS, since: '2025-01-01' }), true)
  assert.equal(filtersSet({ ...NO_FILTERS, contact: 'chat-1' }), true)
})

test('the request carries only what is set', () => {
  assert.deepEqual(filterParams(NO_FILTERS, true), {})
  assert.deepEqual(
    filterParams(
      { contact: 'chat-1', from: 'them', kind: 'group', attachments: 'image', since: '2025-01-01', until: '2025-02-01', order: 'oldest' },
      true,
    ),
    { contact: 'chat-1', from: 'them', kind: 'group', attachments: 'image', since: '2025-01-01', until: '2025-02-01', order: 'oldest' },
  )
})

test('best match needs words: with none, the newest come first', () => {
  assert.deepEqual(filterParams({ ...NO_FILTERS, order: 'relevance' }, true), { order: 'relevance' })
  assert.deepEqual(filterParams({ ...NO_FILTERS, order: 'relevance' }, false), {})
})

test('two searches with other filters have other keys', () => {
  assert.notEqual(filtersKey(NO_FILTERS), filtersKey({ ...NO_FILTERS, kind: 'direct' }))
  assert.notEqual(filtersKey(NO_FILTERS), filtersKey({ ...NO_FILTERS, order: 'oldest' }))
  assert.equal(filtersKey({ ...NO_FILTERS }), filtersKey(NO_FILTERS))
})

test('the status names which part of the matches is shown', () => {
  assert.equal(shownWord({ ...NO_FILTERS, order: 'newest' }, true), 'newest')
  assert.equal(shownWord({ ...NO_FILTERS, order: 'oldest' }, true), 'oldest')
  assert.equal(shownWord({ ...NO_FILTERS, order: 'relevance' }, true), 'best')
  assert.equal(shownWord({ ...NO_FILTERS, order: 'relevance' }, false), 'newest')
})
