import { recallApi } from './api'
import type { AskResponse } from '../types'

// Module-level chat store. Lives outside React so an in-flight request keeps
// running (and its answer still saves) when the user navigates away from Chat,
// and so deleting a chat mid-response clears its loading state cleanly.

export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  response?: AskResponse
}

export type SavedChat = {
  id: string
  title: string
  scope: string
  messages: ChatMessage[]
  createdAt: number
  updatedAt: number
}

export type AskParams = {
  messagesPath: string
  question: string
  contact: string
  model: string
  limit: number
}

type ChatState = {
  chats: SavedChat[]
  activeChatId: string | null
  pending: Record<string, boolean>
  errors: Record<string, string>
}

const CHATS_KEY = 'recall.chats.v1'

function readChats(): SavedChat[] {
  try {
    const raw = window.localStorage.getItem(CHATS_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as SavedChat[]
    return Array.isArray(parsed) ? parsed : []
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
}

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

  deleteChat(id: string) {
    const chats = state.chats.filter((chat) => chat.id !== id)
    persist(chats)
    set({
      chats,
      pending: omitKey(state.pending, id),
      errors: omitKey(state.errors, id),
      activeChatId: state.activeChatId === id ? null : state.activeChatId,
    })
  },

  // Append the question, run the request, and save the answer — all independent
  // of any React component, so it completes even after navigating away.
  async send(question: string, scope: string, ask: AskParams) {
    const text = question.trim()
    if (!text) return
    const now = Date.now()
    const userMessage: ChatMessage = { id: makeId('user'), role: 'user', content: text }

    let chatId = state.activeChatId
    // prior turns let the backend resolve follow-ups ("what about her?")
    const priorChat = chatId ? state.chats.find((chat) => chat.id === chatId) : null
    const history = (priorChat?.messages ?? [])
      .slice(-6)
      .map((message) => ({ role: message.role, content: message.content }))
    if (chatId && chatExists(chatId)) {
      const chats = state.chats.map((chat) =>
        chat.id === chatId
          ? { ...chat, messages: [...chat.messages, userMessage], updatedAt: now }
          : chat,
      )
      persist(chats)
      set({ chats })
    } else {
      chatId = makeId('chat')
      const chat: SavedChat = {
        id: chatId,
        title: text.slice(0, 60),
        scope,
        messages: [userMessage],
        createdAt: now,
        updatedAt: now,
      }
      const chats = [chat, ...state.chats]
      persist(chats)
      set({ chats })
    }

    const targetId = chatId
    set({
      activeChatId: targetId,
      pending: { ...state.pending, [targetId]: true },
      errors: omitKey(state.errors, targetId),
    })

    try {
      const payload = await recallApi.ask({ ...ask, history })
      if (chatExists(targetId)) {
        const assistant: ChatMessage = {
          id: makeId('assistant'),
          role: 'assistant',
          content: payload.answer,
          response: payload,
        }
        const chats = state.chats.map((chat) =>
          chat.id === targetId
            ? { ...chat, messages: [...chat.messages, assistant], updatedAt: Date.now() }
            : chat,
        )
        persist(chats)
        set({ chats, pending: omitKey(state.pending, targetId) })
      } else {
        // chat was deleted mid-flight — discard the answer
        set({ pending: omitKey(state.pending, targetId) })
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unable to query messages.'
      set({
        pending: omitKey(state.pending, targetId),
        errors: chatExists(targetId)
          ? { ...state.errors, [targetId]: message }
          : omitKey(state.errors, targetId),
      })
    }
  },
}
