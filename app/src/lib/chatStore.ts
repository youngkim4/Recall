import { recallApi } from './api'
import { errorSentence } from './apiError'
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
} from './chatMessages'
import { startWork, withProgress, withStatus, workSummary, type Work } from './chatWork'

export type { ChatMessage, SavedChat } from './chatMessages'
export type { Work } from './chatWork'

// Module-level chat store. Lives outside React so an in-flight request keeps
// running (and its answer still saves) when the user navigates away from Chat,
// and so deleting a chat mid-response clears its loading state cleanly.

export type AskParams = {
  messagesPath: string
  question: string
  contact: string
  model: string
  // one of the model's reasoning levels; '' or nothing leaves it to the server
  reasoningEffort?: string
  // not sent: the server decides how much it reads
  limit?: number
}

type ChatState = {
  chats: SavedChat[]
  activeChatId: string | null
  pending: Record<string, boolean>
  errors: Record<string, string>
  // what the server is doing for each chat that waits on an answer, until the answer starts
  work: Record<string, Work>
}

// A question that is being answered: the id the server knows it by, and the
// way to end the request from this side.
type Running = { requestId: string; controller: AbortController }

const CHATS_KEY = 'recall.chats.v1'
const NO_ANSWER = 'Recall did not answer. Ask again.'

function readChats(): SavedChat[] {
  try {
    return readSavedChats(window.localStorage.getItem(CHATS_KEY))
  } catch {
    return []
  }
}

function persist(chats: SavedChat[]) {
  try {
    window.localStorage.setItem(CHATS_KEY, JSON.stringify(chats))
  } catch {
    // Chat history is best-effort; the app still works without storage.
  }
}

function makeId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

// what the server knows a question by; a page that is not secure has no randomUUID
function makeRequestId() {
  return globalThis.crypto?.randomUUID?.() ?? makeId('ask')
}

function omitKey<T>(record: Record<string, T>, key: string): Record<string, T> {
  const next = { ...record }
  delete next[key]
  return next
}

let state: ChatState = {
  chats: readChats(),
  activeChatId: null,
  pending: {},
  errors: {},
  work: {},
}

// the question that each chat is waiting on, by the id of the chat
let running: Record<string, Running> = {}

const listeners = new Set<() => void>()

function set(patch: Partial<ChatState>) {
  state = { ...state, ...patch }
  for (const listener of listeners) listener()
}

function chatExists(id: string): boolean {
  return state.chats.some((chat) => chat.id === id)
}

export const chatStore = {
  subscribe(listener: () => void) {
    listeners.add(listener)
    return () => {
      listeners.delete(listener)
    }
  },

  getState(): ChatState {
    return state
  },

  setActive(id: string | null) {
    set({ activeChatId: id })
  },

  // Ends the question that a chat is waiting on: the request is given up
  // here and the server is told to stop. What was written so far stays, and
  // is marked as stopped. A chat that waits on nothing is left as it is.
  stop(chatId: string) {
    const run = running[chatId]
    if (!run) return
    running = omitKey(running, chatId)
    run.controller.abort()
    // the request is over on this side whatever the server answers
    recallApi.cancelAsk(run.requestId).catch(() => undefined)
    const chats = inChat(state.chats, chatId, (messages) => settled(messages, true))
    persist(chats)
    set({ chats, pending: omitKey(state.pending, chatId), work: omitKey(state.work, chatId) })
  },

  // a chat deleted while it waits on an answer ends that question first
  deleteChat(id: string) {
    chatStore.stop(id)
    const chats = state.chats.filter((chat) => chat.id !== id)
    persist(chats)
    set({
      chats,
      pending: omitKey(state.pending, id),
      errors: omitKey(state.errors, id),
      work: omitKey(state.work, id),
      activeChatId: state.activeChatId === id ? null : state.activeChatId,
    })
  },

  // Undo for deleteChat: put a removed chat back where it was. An answer that
  // was still streaming when the chat was deleted is gone, so any partial
  // message stops being marked as streaming.
  restoreChat(chat: SavedChat, index: number) {
    if (chatExists(chat.id)) return
    const restored: SavedChat = { ...chat, messages: settled(chat.messages) }
    const at = Math.max(0, Math.min(index, state.chats.length))
    const chats = [...state.chats.slice(0, at), restored, ...state.chats.slice(at)]
    persist(chats)
    set({ chats })
  },

  // Append the question, run the request, and save the answer — all independent
  // of any React component, so it completes even after navigating away.
  // `scope` is the conversation that the question is asked of, '' for
  // everything. It stays on the question and goes back with the history.
  async send(question: string, scope: string, ask: AskParams) {
    const text = question.trim()
    if (!text) return
    const prior = state.chats.find((chat) => chat.id === state.activeChatId) ?? null
    // a question asked before the last answer ended takes its place
    if (prior) chatStore.stop(prior.id)

    const now = Date.now()
    const userMessage: ChatMessage = { id: makeId('user'), role: 'user', content: text, scope }
    // prior turns let the backend resolve follow-ups ("what about her?")
    const history = historyRows(state.chats.find((chat) => chat.id === prior?.id)?.messages ?? [])
    const targetId = prior?.id ?? makeId('chat')
    const started: SavedChat = {
      id: targetId,
      title: text.slice(0, 60),
      scope,
      messages: [userMessage],
      createdAt: now,
      updatedAt: now,
    }
    const chats = prior
      ? inChat(state.chats, targetId, (messages) => [...messages, userMessage], now)
      : [started, ...state.chats]
    persist(chats)

    const run: Running = { requestId: makeRequestId(), controller: new AbortController() }
    running = { ...running, [targetId]: run }
    set({
      chats,
      activeChatId: targetId,
      pending: { ...state.pending, [targetId]: true },
      errors: omitKey(state.errors, targetId),
      work: { ...state.work, [targetId]: startWork(now) },
    })
    // the steps of this question as they are now; they stay until the answer
    // is whole, since a reset can take the streamed text away and work goes on
    const workNow = () => state.work[targetId] ?? startWork(now)
    const changeWork = (change: (work: Work) => Work) =>
      set({ work: { ...state.work, [targetId]: change(workNow()) } })

    // False once the question was stopped, its chat deleted, or another
    // question asked in its place: what still arrives for it changes nothing.
    const mine = () => running[targetId] === run
    // the answer has one id from its first streamed piece to its whole text
    const answerId = makeId('assistant')

    try {
      const payload = await recallApi.askStream(
        {
          messagesPath: ask.messagesPath,
          question: text,
          contact: ask.contact,
          model: ask.model,
          reasoningEffort: ask.reasoningEffort,
          history,
          requestId: run.requestId,
        },
        {
          onStatus: (said) => {
            if (mine()) changeWork((work) => withStatus(work, said))
          },
          onProgress: (part) => {
            if (mine()) changeWork((work) => withProgress(work, part))
          },
          // the first piece opens the answer with the work so far, "Worked for 6s"
          onDelta: (delta) => {
            if (!mine()) return
            const work = workSummary(workNow(), Date.now())
            set({ chats: inChat(state.chats, targetId, (messages) => withDelta(messages, answerId, delta, work)) })
          },
          onReset: () => {
            if (mine()) set({ chats: inChat(state.chats, targetId, (messages) => withoutMessage(messages, answerId)) })
          },
        },
        run.controller.signal,
      )
      if (!mine()) return
      running = omitKey(running, targetId)
      const work = workSummary(workNow(), Date.now())
      const answered = inChat(
        state.chats,
        targetId,
        (messages) => withAnswer(messages, answerId, payload, work),
        Date.now(),
      )
      persist(answered)
      set({
        chats: answered,
        pending: omitKey(state.pending, targetId),
        work: omitKey(state.work, targetId),
      })
    } catch (error) {
      // stopped or deleted: stop() already left the chat in order
      if (!mine()) return
      running = omitKey(running, targetId)
      // keep any partial text, but stop marking it as streaming
      const ended = inChat(state.chats, targetId, settled)
      persist(ended)
      set({
        chats: ended,
        pending: omitKey(state.pending, targetId),
        work: omitKey(state.work, targetId),
        // the server's sentence, as it is
        errors: { ...state.errors, [targetId]: errorSentence(error, NO_ANSWER) },
      })
    }
  },
}
