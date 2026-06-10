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
  streaming?: boolean
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
  status: Record<string, string>
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
  status: {},
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
      status: omitKey(state.status, id),
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
      status: { ...state.status, [targetId]: 'Thinking…' },
    })

    // created on the first streamed token, then grows in place
    let assistantId: string | null = null

    const appendDelta = (delta: string) => {
      if (!chatExists(targetId)) return // chat deleted mid-stream
      if (!assistantId) {
        const id = makeId('assistant')
        assistantId = id
        const opener: ChatMessage = { id, role: 'assistant', content: delta, streaming: true }
        const chats = state.chats.map((chat) =>
          chat.id === targetId ? { ...chat, messages: [...chat.messages, opener] } : chat,
        )
        set({ chats, status: omitKey(state.status, targetId) })
        return
      }
      const chats = state.chats.map((chat) =>
        chat.id === targetId
          ? {
              ...chat,
              messages: chat.messages.map((message) =>
                message.id === assistantId
                  ? { ...message, content: message.content + delta }
                  : message,
              ),
            }
          : chat,
      )
      set({ chats })
    }

    try {
      const payload = await recallApi.askStream(
        { ...ask, history },
        {
          onStatus: (text) => {
            if (chatExists(targetId) && state.pending[targetId]) {
              set({ status: { ...state.status, [targetId]: text } })
            }
          },
          onDelta: appendDelta,
        },
      )
      if (chatExists(targetId)) {
        // If the model failed AFTER text streamed, the server falls back to a
        // keyword summary (mode 'local'). Never replace prose the user has
        // already read with that — keep the streamed text, trim the citation
        // dump, and let the note explain.
        const streamedContent = assistantId
          ? state.chats
              .find((chat) => chat.id === targetId)
              ?.messages.find((message) => message.id === assistantId)?.content ?? ''
          : ''
        const keepStreamed = payload.mode === 'local' && streamedContent.length > 80
        const final: ChatMessage = {
          id: assistantId ?? makeId('assistant'),
          role: 'assistant',
          content: keepStreamed ? streamedContent : payload.answer,
          response: keepStreamed
            ? { ...payload, mode: 'ai', citations: payload.citations.slice(0, 6) }
            : payload,
        }
        const chats = state.chats.map((chat) =>
          chat.id === targetId
            ? {
                ...chat,
                updatedAt: Date.now(),
                messages: assistantId
                  ? chat.messages.map((message) => (message.id === assistantId ? final : message))
                  : [...chat.messages, final],
              }
            : chat,
        )
        persist(chats)
        set({
          chats,
          pending: omitKey(state.pending, targetId),
          status: omitKey(state.status, targetId),
        })
      } else {
        // chat was deleted mid-flight — discard the answer
        set({
          pending: omitKey(state.pending, targetId),
          status: omitKey(state.status, targetId),
        })
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unable to query messages.'
      // keep any partial text, but stop marking it as streaming
      const chats = assistantId
        ? state.chats.map((chat) =>
            chat.id === targetId
              ? {
                  ...chat,
                  messages: chat.messages.map((msg) =>
                    msg.id === assistantId ? { ...msg, streaming: undefined } : msg,
                  ),
                }
              : chat,
          )
        : state.chats
      if (assistantId) persist(chats)
      set({
        chats,
        pending: omitKey(state.pending, targetId),
        status: omitKey(state.status, targetId),
        errors: chatExists(targetId)
          ? { ...state.errors, [targetId]: message }
          : omitKey(state.errors, targetId),
      })
    }
  },
}
