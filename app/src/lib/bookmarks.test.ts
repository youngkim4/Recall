import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { SearchResult } from '../types'
import { bookmarkId, readBookmarks, toggleBookmark } from './bookmarks.ts'
import { fakeStorage, noPage, pageWithNoStorage, pageWithStorage } from './fakeStorage.test.ts'

const STORE_KEY = 'recall.bookmarks.v1'

const message = (fields: Partial<SearchResult> = {}): SearchResult => ({
  chatId: '+15550100199',
  timestamp: '2025-03-20 09:00:00',
  text: 'on my way',
  ...fields,
})

test('a message is known by its conversation, its time and its words, never by its id', () => {
  // one message has three ids in its life: a number, legacy:<n>, an Apple guid
  const first = message({ messageId: '4471' })

  assert.equal(bookmarkId(first), bookmarkId({ ...first, messageId: 'legacy:4471' }))
  assert.equal(bookmarkId(first), bookmarkId({ ...first, messageId: '8A6F0C2E-1B7D-4E0A-9C55-3F2D1A7B9E10' }))
  assert.equal(bookmarkId(first), bookmarkId(message()))
  assert.notEqual(bookmarkId(first), bookmarkId({ ...first, text: 'running late' }))
})

test('a message saved by an older build is the same message in a new search', () => {
  pageWithStorage(fakeStorage({ [STORE_KEY]: JSON.stringify([message({ messageId: '4471' })]) }))
  const found = message({ messageId: '8A6F0C2E-1B7D-4E0A-9C55-3F2D1A7B9E10', displayName: 'Avery Stone' })

  const saved = readBookmarks()

  assert.equal(saved.map(bookmarkId).includes(bookmarkId(found)), true)
  // it is saved once: pressing again takes it back out
  assert.deepEqual(toggleBookmark(saved, found), [])
})

test('an id that an older build kept as a number is read as text', () => {
  const numbered = { ...message(), messageId: 4471 }
  pageWithStorage(fakeStorage({ [STORE_KEY]: JSON.stringify([numbered]) }))

  assert.deepEqual(readBookmarks(), [message({ messageId: '4471' })])
})

test('spaces at the ends of the words do not make another message', () => {
  assert.equal(bookmarkId(message({ text: ' on my way\n' })), bookmarkId(message()))
  assert.notEqual(bookmarkId(message({ text: 'on  my way' })), bookmarkId(message()))
})

test('a message with no time is still told from one with a time', () => {
  const timeless = message({ timestamp: null })

  assert.equal(bookmarkId(timeless), bookmarkId({ ...timeless, messageId: 'legacy:9' }))
  assert.notEqual(bookmarkId(timeless), bookmarkId(message()))
})

test('two messages with no ids that begin alike are still two messages', () => {
  const start = 'see you at the station at '
  const one = message({ text: `${start}eight` })
  const other = message({ text: `${start}nine` })

  assert.notEqual(bookmarkId(one), bookmarkId(other))
  assert.equal(bookmarkId(one), bookmarkId({ ...one }))
})

test('the chat and the moment are part of what a message is known by', () => {
  assert.notEqual(bookmarkId(message()), bookmarkId(message({ chatId: 'chat4471' })))
  assert.notEqual(bookmarkId(message()), bookmarkId(message({ timestamp: '2025-03-20 09:00:01' })))
  // an id is not mistaken for the words of another message
  assert.notEqual(bookmarkId(message({ messageId: 'on my way', text: '' })), bookmarkId(message()))
})

test('saving puts a message first, and saving it again takes it back out', () => {
  const storage = pageWithStorage(fakeStorage())
  const older = message({ messageId: 'm-1' })
  const newer = message({ messageId: 'm-2', text: 'running late' })

  const one = toggleBookmark([], older)
  const two = toggleBookmark(one, newer)

  assert.deepEqual(two, [newer, older])
  assert.deepEqual(JSON.parse(storage.dump()[STORE_KEY]), [newer, older])
  assert.deepEqual(readBookmarks(), [newer, older])
  assert.deepEqual(toggleBookmark(two, older), [newer])
  // the list it was given is left as it was
  assert.deepEqual(two, [newer, older])
})

test('taking out one of two messages that begin alike leaves the other', () => {
  pageWithStorage(fakeStorage())
  const start = 'see you at the station at '
  const one = message({ text: `${start}eight` })
  const other = message({ text: `${start}nine` })

  assert.deepEqual(toggleBookmark([one, other], other), [one])
})

test('the oldest saved messages give way past a hundred', () => {
  pageWithStorage(fakeStorage())
  const many = Array.from({ length: 100 }, (_, index) => message({ messageId: `m-${index}`, text: `message ${index}` }))

  const next = toggleBookmark(many, message({ messageId: 'new', text: 'a new message' }))

  assert.equal(next.length, 100)
  assert.equal(next[0].messageId, 'new')
  assert.equal(next[99].messageId, 'm-98')
})

test('what was saved is read back, and anything that is not a message is left out', () => {
  const whole = message({ messageId: 'm-1', displayName: 'Avery Stone', isFromMe: 1 })
  const timeless = { chatId: 'chat4471', text: 'photo', timestamp: null }
  pageWithStorage(
    fakeStorage({
      [STORE_KEY]: JSON.stringify([
        null,
        7,
        'words',
        [],
        { chatId: '+15550100142' },
        { text: 'no chat' },
        { chatId: '+15550100142', text: 12 },
        { chatId: '+15550100142', text: 'a name that is not text', displayName: 9 },
        { chatId: '+15550100142', text: 'a time that is not text', timestamp: {} },
        whole,
        timeless,
      ]),
    }),
  )

  assert.deepEqual(readBookmarks(), [whole, timeless])
  assert.doesNotThrow(() => readBookmarks().map(bookmarkId))
})

test('a store that does not hold a list of messages reads as nothing saved', () => {
  ;['{{{', '{"a":1}', '7', 'null', '"text"'].forEach((raw) => {
    pageWithStorage(fakeStorage({ [STORE_KEY]: raw }))

    assert.deepEqual(readBookmarks(), [], raw)
  })
})

test('no more than a hundred saved messages are read', () => {
  const many = Array.from({ length: 140 }, (_, index) => message({ messageId: `m-${index}` }))
  pageWithStorage(fakeStorage({ [STORE_KEY]: JSON.stringify(many) }))

  assert.equal(readBookmarks().length, 100)
  assert.equal(readBookmarks()[0].messageId, 'm-0')
})

test('where nothing can be kept, saving still works for the session', () => {
  const first = message({ messageId: 'm-1' })

  pageWithNoStorage()
  assert.deepEqual(readBookmarks(), [])
  assert.deepEqual(toggleBookmark([], first), [first])

  pageWithStorage(fakeStorage({}, { broken: true }))
  assert.deepEqual(readBookmarks(), [])
  assert.deepEqual(toggleBookmark([], first), [first])

  noPage()
  assert.deepEqual(readBookmarks(), [])
  assert.deepEqual(toggleBookmark([first], first), [])
})
