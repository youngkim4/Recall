import { useEffect, useMemo, useRef, useState } from 'react'
import { SparkIcon } from './Icons'
import { recallApi } from '../lib/api'
import { contactTitle, isOutbound, prettyContactId, shortDateTime } from '../lib/format'
import { modelLabel } from '../lib/models'
import type { AskResponse, Contact, Defaults } from '../types'

type AskViewProps = {
  defaults: Defaults | null
  model: string
  onModelChange: (model: string) => void
  contacts: Contact[]
  onSelectContact: (contact: Contact) => void
}

type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  response?: AskResponse
}

type SavedChat = {
  id: string
  title: string
  scope: string
  messages: ChatMessage[]
  createdAt: number
  updatedAt: number
}

const CHATS_KEY = 'recall.chats.v1'

const starterPrompts = [
  'What changed between us over time?',
  'Find moments where trust came up.',
  'What did we talk about around travel?',
  'Summarize the most emotional messages.',
]

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

function storeChats(chats: SavedChat[]) {
  try {
    window.localStorage.setItem(CHATS_KEY, JSON.stringify(chats))
  } catch {
    // Chat history is best-effort; the app still works without storage.
  }
}

function makeId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

function nowMs() {
  return Date.now()
}

function relativeTime(ms: number) {
  const diff = Date.now() - ms
  const minutes = Math.floor(diff / 60000)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days < 7) return `${days}d ago`
  return new Date(ms).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

export function AskView({ defaults, model, onModelChange, contacts, onSelectContact }: AskViewProps) {
  const [chats, setChats] = useState<SavedChat[]>(() => readChats())
  const [activeChatId, setActiveChatId] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [scope, setScope] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const threadRef = useRef<HTMLDivElement | null>(null)

  const activeChat = useMemo(
    () => chats.find((chat) => chat.id === activeChatId) ?? null,
    [chats, activeChatId],
  )
  const messages = activeChat?.messages ?? []

  useEffect(() => {
    threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages.length, loading])

  function newChat() {
    setActiveChatId(null)
    setDraft('')
    setError('')
  }

  function openChat(id: string) {
    const chat = chats.find((item) => item.id === id)
    setActiveChatId(id)
    setScope(chat?.scope ?? '')
    setDraft('')
    setError('')
  }

  function deleteChat(id: string) {
    setChats((current) => {
      const next = current.filter((chat) => chat.id !== id)
      storeChats(next)
      return next
    })
    if (activeChatId === id) setActiveChatId(null)
  }

  async function submit(nextQuestion = draft) {
    const question = nextQuestion.trim()
    if (!defaults?.messagesPath || !question || loading) return

    const userMessage: ChatMessage = { id: makeId('user'), role: 'user', content: question }
    const now = nowMs()
    const chatId = activeChat ? activeChat.id : makeId('chat')

    setChats((current) => {
      const exists = current.some((chat) => chat.id === chatId)
      const next = exists
        ? current.map((chat) =>
            chat.id === chatId
              ? { ...chat, messages: [...chat.messages, userMessage], updatedAt: now }
              : chat,
          )
        : [
            {
              id: chatId,
              title: question.slice(0, 60),
              scope,
              messages: [userMessage],
              createdAt: now,
              updatedAt: now,
            },
            ...current,
          ]
      storeChats(next)
      return next
    })
    setActiveChatId(chatId)
    setDraft('')
    setLoading(true)
    setError('')

    try {
      const payload = await recallApi.ask({
        messagesPath: defaults.messagesPath,
        question,
        contact: scope,
        model,
        limit: 10,
      })
      const assistantMessage: ChatMessage = {
        id: makeId('assistant'),
        role: 'assistant',
        content: payload.answer,
        response: payload,
      }
      setChats((current) => {
        const next = current.map((chat) =>
          chat.id === chatId
            ? { ...chat, messages: [...chat.messages, assistantMessage], updatedAt: nowMs() }
            : chat,
        )
        storeChats(next)
        return next
      })
    } catch (apiError) {
      setError(apiError instanceof Error ? apiError.message : 'Unable to query messages.')
    } finally {
      setLoading(false)
    }
  }

  function openCitation(chatId: string) {
    const contact = contacts.find((item) => item.chat_id === chatId)
    if (contact) onSelectContact(contact)
  }

  return (
    <div className="ask-layout">
      <aside className="chat-history" aria-label="Saved chats">
        <button type="button" className="new-chat-button" onClick={newChat}>
          <span aria-hidden>+</span> New chat
        </button>
        <div className="chat-history-list">
          {chats.length ? (
            chats.map((chat) => (
              <div
                key={chat.id}
                className={`chat-history-row ${chat.id === activeChatId ? 'active' : ''}`}
              >
                <button type="button" className="chat-history-open" onClick={() => openChat(chat.id)}>
                  <strong>{chat.title || 'New chat'}</strong>
                  <span>{relativeTime(chat.updatedAt)}</span>
                </button>
                <button
                  type="button"
                  className="chat-history-delete"
                  aria-label="Delete chat"
                  onClick={() => deleteChat(chat.id)}
                >
                  ×
                </button>
              </div>
            ))
          ) : (
            <p className="chat-history-empty">No saved chats yet.</p>
          )}
        </div>
      </aside>

      <section className={`ask-view chat-view ${messages.length ? '' : 'is-empty'}`} aria-label="Recall">
        <div className="chat-thread" ref={threadRef}>
          {messages.length ? (
            messages.map((message) => (
              <article key={message.id} className={`chat-message ${message.role}`}>
                <div className="chat-avatar">{message.role === 'user' ? 'Y' : 'R'}</div>
                <div className="chat-bubble">
                  <div className="chat-meta">
                    <strong>{message.role === 'user' ? 'You' : 'Recall'}</strong>
                  </div>
                  <p>{message.content}</p>
                  {message.response ? (
                    <div className="chat-evidence">
                      {message.response.terms.length ? (
                        <div className="term-list">
                          {message.response.terms.map((term) => (
                            <span key={term}>{term}</span>
                          ))}
                        </div>
                      ) : null}
                      {message.response.citations.length ? (
                        <div className="citation-list compact">
                          {message.response.citations.map((citation) => (
                            <button
                              key={`${citation.chatId}-${citation.timestamp}-${citation.text.slice(0, 20)}`}
                              type="button"
                              className="citation-row"
                              onClick={() => openCitation(citation.chatId)}
                            >
                              <span>
                                {citation.displayName || prettyContactId(citation.chatId)}{' '}
                                / {shortDateTime(citation.timestamp)}
                              </span>
                              <p className={isOutbound(citation) ? 'outbound' : ''}>{citation.text || '(empty)'}</p>
                            </button>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              </article>
            ))
          ) : (
            <div className="chat-empty">
              <h2>Chat with your message archive.</h2>
              <div className="prompt-grid">
                {starterPrompts.map((prompt) => (
                  <button key={prompt} type="button" onClick={() => void submit(prompt)}>
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          )}

          {loading ? (
            <article className="chat-message assistant">
              <div className="chat-avatar">R</div>
              <div className="chat-bubble typing">
                <span />
                <span />
                <span />
              </div>
            </article>
          ) : null}
        </div>

        {error ? <div className="inline-error chat-error">{error}</div> : null}

        <form
          className="chat-composer"
          onSubmit={(event) => {
            event.preventDefault()
            void submit()
          }}
        >
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                void submit()
              }
            }}
            placeholder="Message Recall..."
            rows={1}
          />
          <div className="chat-composer-footer">
            <div className="chat-controls">
              <select value={scope} onChange={(event) => setScope(event.target.value)} aria-label="Conversation scope">
                <option value="">All conversations</option>
                {contacts.map((contact) => (
                  <option key={contact.chat_id} value={contact.chat_id}>
                    {contactTitle(contact)}
                  </option>
                ))}
              </select>
              <select value={model} onChange={(event) => onModelChange(event.target.value)} aria-label="Model">
                {(defaults?.models || [model]).map((option) => (
                  <option key={option} value={option}>
                    {modelLabel(option)}
                  </option>
                ))}
              </select>
            </div>
            <button
              type="submit"
              className="chat-send"
              disabled={!defaults?.messagesPath || !draft.trim() || loading}
              aria-label="Send message"
            >
              <SparkIcon className="button-icon" />
            </button>
          </div>
        </form>
      </section>
    </div>
  )
}
