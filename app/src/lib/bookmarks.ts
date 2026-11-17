import type { SearchResult } from '../types'

// Messages the reader saved from a search. They are kept in this browser
// only, newest first, and the oldest give way past the limit.
const BOOKMARKS_KEY = 'recall.bookmarks.v1'
const BOOKMARK_LIMIT = 100

// What tells one message from every other: its conversation, its time and
// its words. Never its id: a message has three ids in its life (a number,
// "legacy:<n>", an Apple guid), and it is the same message under each. It is
// worked out from the message each time, so a message saved by an older
// build is still matched to the same message in a new search. The spaces at
// the ends of the words are not part of it.
export function bookmarkId(message: SearchResult) {
  return [message.chatId, message.timestamp ?? '', message.text.trim()].join('\n')
}

// what a message may leave out, and must have as text when it does not
const LABELS = ['timestamp', 'displayName', 'senderName', 'sender', 'attachment']
const textOrMissing = (value: unknown) => value === undefined || value === null || typeof value === 'string'
// an older build kept the id as the number it then was
const idOrMissing = (value: unknown) => textOrMissing(value) || typeof value === 'number'

// A stored value that can be shown as a message again: its chat and its words
// are text, and so is every label it carries.
function isSavedMessage(value: unknown): value is SearchResult {
  if (!value || typeof value !== 'object') return false
  const message = value as Record<string, unknown>
  return (
    typeof message.chatId === 'string' &&
    typeof message.text === 'string' &&
    idOrMissing(message.messageId) &&
    LABELS.every((label) => textOrMissing(message[label]))
  )
}

// the id of a message is text, whatever the build that saved it wrote
function withTextId(message: SearchResult): SearchResult {
  return typeof message.messageId === 'number' ? { ...message, messageId: String(message.messageId) } : message
}

export function readBookmarks(): SearchResult[] {
  try {
    const stored = window.localStorage.getItem(BOOKMARKS_KEY)
    const parsed: unknown = stored ? JSON.parse(stored) : []
    return Array.isArray(parsed) ? parsed.filter(isSavedMessage).map(withTextId).slice(0, BOOKMARK_LIMIT) : []
  } catch {
    return []
  }
}

function storeBookmarks(bookmarks: SearchResult[]) {
  try {
    window.localStorage.setItem(BOOKMARKS_KEY, JSON.stringify(bookmarks))
  } catch {
    // saved messages are a convenience; a full or closed store only means they do not persist
  }
}

// Saves a message, or takes it back out when it is already saved. Returns the new list.
export function toggleBookmark(bookmarks: SearchResult[], message: SearchResult) {
  const id = bookmarkId(message)
  const saved = bookmarks.some((item) => bookmarkId(item) === id)
  const next = saved
    ? bookmarks.filter((item) => bookmarkId(item) !== id)
    : [message, ...bookmarks].slice(0, BOOKMARK_LIMIT)
  storeBookmarks(next)
  return next
}
