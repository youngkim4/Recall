// The messages of a chat and the changes an answer makes to them. Nothing
// here keeps state or talks to the server: chatStore.ts does both.
import type { AskResponse } from '../types'
import type { WorkSummary } from './chatWork.ts'

export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  // On a question: the conversation it was asked of, '' for everything. A
  // question saved by an older build has none.
  scope?: string
  // what the server said of a whole answer, kept as it came
  response?: AskResponse
  streaming?: boolean
  // the reader stopped the answer before it ended
  stopped?: boolean
  // On an answer: how long the work before it took and its steps, shown as
  // "Worked for 6s" over it. An answer of an older build has none.
  work?: WorkSummary
}

export type SavedChat = {
  id: string
  title: string
  scope: string
  messages: ChatMessage[]
  createdAt: number
  updatedAt: number
}

// an earlier turn, as /api/ask reads it
export type HistoryRow = { role: 'user' | 'assistant'; content: string; contact?: string }

// how many earlier turns go with a question, so a follow-up can be understood
const HISTORY_TURNS = 6

// The last turns of a chat for the server. Each question names the
// conversation it was asked of. One from an older build names none, and an
// answer never does.
export function historyRows(messages: ChatMessage[], turns = HISTORY_TURNS): HistoryRow[] {
  return messages.slice(-turns).map((message) =>
    message.role === 'user' && typeof message.scope === 'string'
      ? { role: message.role, content: message.content, contact: message.scope }
      : { role: message.role, content: message.content },
  )
}

// The next piece of an answer: the first one opens the answer, with the work
// done before it; the others grow it.
export function withDelta(messages: ChatMessage[], id: string, delta: string, work?: WorkSummary): ChatMessage[] {
  if (!messages.some((message) => message.id === id)) {
    const opened: ChatMessage = { id, role: 'assistant', content: delta, streaming: true }
    return [...messages, work ? { ...opened, work } : opened]
  }
  return messages.map((message) => (message.id === id ? { ...message, content: message.content + delta } : message))
}

// The text streamed so far was a note of the model and not the answer: it goes.
export function withoutMessage(messages: ChatMessage[], id: string): ChatMessage[] {
  return messages.filter((message) => message.id !== id)
}

// The whole answer, in the place of what was streamed. It is shown as the
// server gave it, and keeps the work its first piece came with; an answer
// with no streamed text takes the work it is given.
export function withAnswer(messages: ChatMessage[], id: string, payload: AskResponse, work?: WorkSummary): ChatMessage[] {
  const streamed = messages.find((message) => message.id === id)
  const kept = streamed?.work ?? work
  const answer: ChatMessage = { id, role: 'assistant', content: payload.answer, response: payload }
  const whole = kept ? { ...answer, work: kept } : answer
  return streamed ? messages.map((message) => (message.id === id ? whole : message)) : [...messages, whole]
}

// An answer that ended before it was whole keeps its text and is no longer
// being written. `stopped` says that the reader ended it.
export function settled(messages: ChatMessage[], stopped = false): ChatMessage[] {
  return messages.map((message) => {
    if (!message.streaming) return message
    return stopped ? { ...message, streaming: undefined, stopped: true } : { ...message, streaming: undefined }
  })
}

// The chats with the messages of one of them changed. `at` is when, for a
// change that brings the chat up the list.
export function inChat(
  chats: SavedChat[],
  chatId: string,
  change: (messages: ChatMessage[]) => ChatMessage[],
  at?: number,
): SavedChat[] {
  return chats.map((chat) =>
    chat.id === chatId ? { ...chat, messages: change(chat.messages), updatedAt: at ?? chat.updatedAt } : chat,
  )
}

function isSavedChat(value: unknown): value is SavedChat {
  const chat = value as Partial<SavedChat> | null
  return Boolean(chat && typeof chat === 'object' && typeof chat.id === 'string' && Array.isArray(chat.messages))
}

// The chats that the browser kept, a build of any age. Nothing in them is
// still being written: an answer that was, when the page closed, ended there.
export function readSavedChats(raw: string | null): SavedChat[] {
  try {
    const parsed: unknown = raw ? JSON.parse(raw) : []
    if (!Array.isArray(parsed)) return []
    return parsed.filter(isSavedChat).map((chat) => ({ ...chat, messages: settled(chat.messages) }))
  } catch {
    // a store that is damaged holds no chats
    return []
  }
}
