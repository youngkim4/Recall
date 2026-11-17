import assert from 'node:assert/strict'
import { test } from 'node:test'
import { importRequest } from './setupGuide.ts'

test('an import writes the archive file that the app reads', () => {
  const request = importRequest({ archivePath: '/elsewhere/mine.csv', copyPath: '' })

  assert.deepEqual(request, { messagesPath: '/elsewhere/mine.csv', dbPath: undefined, includeContacts: true })
  // what is sent names the archive and no database
  assert.deepEqual(JSON.parse(JSON.stringify(request)), { messagesPath: '/elsewhere/mine.csv', includeContacts: true })
})

test('with no archive file set, the server writes its own', () => {
  for (const archivePath of ['', '   ']) {
    assert.deepEqual(JSON.parse(JSON.stringify(importRequest({ archivePath, copyPath: '' }))), { includeContacts: true })
  }
})

test('a copy is named only when one was chosen, and with the archive file', () => {
  const request = importRequest({ archivePath: '/elsewhere/mine.csv', copyPath: '/copies/chat.db' })

  assert.deepEqual(request, { messagesPath: '/elsewhere/mine.csv', dbPath: '/copies/chat.db', includeContacts: true })
})
