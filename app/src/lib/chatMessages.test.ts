import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { AskResponse } from '../types.ts'
import {
  historyRows,
  inChat,
  readSavedChats,
  settled,
  withAnswer,
  withDelta,
  withoutMessage,
  type ChatMessage,
  type SavedChat,
} from './chatMessages.ts'

const asked = (content: string, scope?: string): ChatMessage =>
  scope === undefined ? { id: `user-${content}`, role: 'user', content } : { id: `user-${content}`, role: 'user', content, scope }
const answered = (content: string, fields: Partial<ChatMessage> = {}): ChatMessage => ({
  id: `assistant-${content}`,
  role: 'assistant',
  content,
  ...fields,
})

const answer = (fields: Partial<AskResponse> = {}): AskResponse => ({
  question: 'When did we last talk about the lake?',
  terms: [],
  answer: 'In August.',
  mode: 'ai',
  citations: [],
  ...fields,
})

const chat = (id: string, messages: ChatMessage[]): SavedChat => ({
  id,
  title: 'A chat',
  scope: '',
  messages,
  createdAt: 1,
  updatedAt: 1,
})

test('each question goes back to the server with the conversation it was asked of', () => {
  const rows = historyRows([asked('about her', '+15550100199'), answered('In August.'), asked('and everyone?', '')])

  assert.deepEqual(rows, [
    { role: 'user', content: 'about her', contact: '+15550100199' },
    { role: 'assistant', content: 'In August.' },
    // '' is everything, and it is said
    { role: 'user', content: 'and everyone?', contact: '' },
  ])
})

test('a question of an older build has no conversation to send', () => {
  const rows = historyRows([asked('from before'), answered('An answer.')])

  assert.deepEqual(rows, [
    { role: 'user', content: 'from before' },
    { role: 'assistant', content: 'An answer.' },
  ])
  assert.equal('contact' in rows[0], false)
})

test('an answer never carries a conversation, and only the last turns are sent', () => {
  const many = Array.from({ length: 5 }, (_, index) => [asked(`q${index}`, 'chat4471'), answered(`a${index}`, { scope: 'chat4471' })]).flat()

  const rows = historyRows(many)

  assert.equal(rows.length, 6)
  assert.deepEqual(rows.map((row) => row.content), ['q2', 'a2', 'q3', 'a3', 'q4', 'a4'])
  assert.equal(rows.some((row) => row.role === 'assistant' && 'contact' in row), false)
  assert.deepEqual(historyRows(many, 2).map((row) => row.content), ['q4', 'a4'])
  assert.deepEqual(historyRows([]), [])
})

test('the first piece of an answer opens it, and the next ones grow it', () => {
  const start = [asked('q', '')]

  const opened = withDelta(start, 'assistant-1', 'In ')
  const grown = withDelta(opened, 'assistant-1', 'August.')

  assert.deepEqual(opened[1], { id: 'assistant-1', role: 'assistant', content: 'In ', streaming: true })
  assert.deepEqual(grown[1], { id: 'assistant-1', role: 'assistant', content: 'In August.', streaming: true })
  // the lists that were given are left as they were
  assert.equal(start.length, 1)
  assert.equal(opened[1].content, 'In ')
})

test('text that was not the answer is dropped, and the answer starts again', () => {
  const noted = withDelta([asked('q', '')], 'assistant-1', 'Let me look at the messages first.')

  const cleared = withoutMessage(noted, 'assistant-1')

  assert.deepEqual(cleared, [asked('q', '')])
  assert.deepEqual(withDelta(cleared, 'assistant-1', 'In August.')[1].content, 'In August.')
  // nothing streamed yet: nothing to drop
  assert.deepEqual(withoutMessage([asked('q', '')], 'assistant-1'), [asked('q', '')])
})

test('the whole answer takes the place of what was streamed', () => {
  const streamed = withDelta([asked('q', '')], 'assistant-1', 'In Aug')
  const payload = answer({ unverifiedQuotes: ['"see you at the lake"'], truncated: true, cancelled: false })

  const done = withAnswer(streamed, 'assistant-1', payload)

  assert.deepEqual(done[1], { id: 'assistant-1', role: 'assistant', content: 'In August.', response: payload })
  // what the server said of the answer is kept as it came
  assert.deepEqual(done[1].response?.unverifiedQuotes, ['"see you at the lake"'])
  assert.equal(done[1].response?.truncated, true)
  assert.equal(done[1].response?.cancelled, false)
  assert.equal(done.length, 2)
})

test('an answer that came with no streamed text is added', () => {
  const done = withAnswer([asked('q', '')], 'assistant-1', answer())

  assert.deepEqual(done[1], { id: 'assistant-1', role: 'assistant', content: 'In August.', response: answer() })
})

test('a plain answer is not rewritten, whatever was streamed before it', () => {
  const long = 'x'.repeat(200)
  const streamed = withDelta([asked('q', '')], 'assistant-1', long)
  const payload = answer({ mode: 'local', answer: 'Matches for your words.' })

  const done = withAnswer(streamed, 'assistant-1', payload)

  assert.equal(done[1].content, 'Matches for your words.')
  assert.equal(done[1].response?.mode, 'local')
})

test('an answer that ended early keeps its text and stops being written', () => {
  const streamed = withDelta([asked('q', '')], 'assistant-1', 'In Aug')

  assert.deepEqual(settled(streamed)[1], { id: 'assistant-1', role: 'assistant', content: 'In Aug', streaming: undefined })
  // one that the reader stopped says so
  assert.deepEqual(settled(streamed, true)[1], {
    id: 'assistant-1',
    role: 'assistant',
    content: 'In Aug',
    streaming: undefined,
    stopped: true,
  })
  // an answer that was whole is left as it is
  const whole = [asked('q', ''), answered('In August.')]
  assert.deepEqual(settled(whole, true), whole)
})

test('a change is made in one chat and the others are left', () => {
  const chats = [chat('a', [asked('q', '')]), chat('b', [])]

  const next = inChat(chats, 'a', (messages) => withDelta(messages, 'assistant-1', 'x'), 99)

  assert.equal(next[0].messages.length, 2)
  assert.equal(next[0].updatedAt, 99)
  assert.equal(next[1], chats[1])
  assert.equal(chats[0].messages.length, 1)
  // with no time given, the chat keeps the time it had
  assert.equal(inChat(chats, 'a', (messages) => messages)[0].updatedAt, 1)
  assert.deepEqual(inChat(chats, 'gone', () => []), chats)
})

test('the chats of an older build are read, and no answer in them is still being written', () => {
  const older = [
    {
      id: 'chat-1',
      title: 'The lake',
      scope: '+15550100199',
      createdAt: 1,
      updatedAt: 2,
      messages: [
        { id: 'user-1', role: 'user', content: 'When did we last talk about the lake?' },
        { id: 'assistant-1', role: 'assistant', content: 'In Aug', streaming: true },
      ],
    },
  ]

  const read = readSavedChats(JSON.stringify(older))

  assert.equal(read.length, 1)
  assert.equal(read[0].messages[0].scope, undefined)
  assert.equal(read[0].messages[1].streaming, undefined)
  assert.deepEqual(historyRows(read[0].messages)[0], { role: 'user', content: 'When did we last talk about the lake?' })
})

test('a store that does not hold chats reads as no chats', () => {
  assert.deepEqual(readSavedChats(null), [])
  assert.deepEqual(readSavedChats(''), [])
  assert.deepEqual(readSavedChats('{{{'), [])
  assert.deepEqual(readSavedChats('{"a":1}'), [])
  assert.deepEqual(readSavedChats('[null, 7, "text", {"id": "chat-1"}, {"id": 3, "messages": []}]'), [])
})

test('an answer keeps how long the work before it took, from its first piece to its whole text', () => {
  const work = { seconds: 4, steps: ['Reading your archive'] }
  const opened = withDelta([asked('q', '')], 'assistant-1', 'In ', work)
  assert.deepEqual(opened[1].work, work)
  // a later piece does not change it
  const grown = withDelta(opened, 'assistant-1', 'August.', { seconds: 9, steps: [] })
  assert.deepEqual(grown[1].work, work)
  assert.deepEqual(withAnswer(grown, 'assistant-1', answer())[1].work, work)
})

test('an answer with no streamed text takes the work it is given', () => {
  const work = { seconds: 2, steps: ['Searching'] }
  assert.deepEqual(withAnswer([asked('q', '')], 'assistant-1', answer(), work)[1].work, work)
  assert.equal(withAnswer([asked('q', '')], 'assistant-1', answer())[1].work, undefined)
})
