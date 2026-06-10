import { useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from 'react'
import { SparkIcon } from './Icons'
import { contactLabel, contactTitle, isOutbound, shortDateTime } from '../lib/format'
import { modelLabel } from '../lib/models'
import { chatStore } from '../lib/chatStore'
import type { Contact, Defaults } from '../types'

type AskViewProps = {
  defaults: Defaults | null
  model: string
  onModelChange: (model: string) => void
  contacts: Contact[]
  onSelectContact: (contact: Contact) => void
}

const starterPrompts = [
  'What changed between us over time?',
  'Find moments where trust came up.',
  'What did we talk about around travel?',
  'Summarize the most emotional messages.',
]

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

const INLINE_PATTERN = /(\*\*[^*]+\*\*|\[\d+\])/g

// minimal inline formatter: **bold** and [n] citation markers -> react nodes
function renderInline(text: string, keyBase: string): ReactNode[] {
  return text.split(INLINE_PATTERN).map((part, index) => {
    if (!part) return null
    const bold = /^\*\*([^*]+)\*\*$/.exec(part)
    if (bold) return <strong key={`${keyBase}-b${index}`}>{bold[1]}</strong>
    const cite = /^\[(\d+)\]$/.exec(part)
    if (cite)
      return (
        <sup key={`${keyBase}-c${index}`} className="cite-ref">
          {cite[1]}
        </sup>
      )
    return <span key={`${keyBase}-t${index}`}>{part}</span>
  })
}

// render the assistant answer as conversational prose: paragraphs, soft
// line breaks, simple bullet lists, inline bold and citation markers
function AnswerBody({ text }: { text: string }) {
  const blocks = text.trim().split(/\n{2,}/).filter(Boolean)
  return (
    <div className="answer-body">
      {blocks.map((block, bi) => {
        const lines = block.split('\n')
        const isList = lines.length > 0 && lines.every((line) => /^\s*[-*]\s+/.test(line))
        if (isList) {
          return (
            <ul key={`blk-${bi}`} className="answer-list">
              {lines.map((line, li) => (
                <li key={`blk-${bi}-li-${li}`}>
                  {renderInline(line.replace(/^\s*[-*]\s+/, ''), `blk-${bi}-${li}`)}
                </li>
              ))}
            </ul>
          )
        }
        return (
          <p key={`blk-${bi}`}>
            {lines.map((line, li) => (
              <span key={`blk-${bi}-ln-${li}`}>
                {renderInline(line, `blk-${bi}-${li}`)}
                {li < lines.length - 1 ? <br /> : null}
              </span>
            ))}
          </p>
        )
      })}
    </div>
  )
}

export function AskView({ defaults, model, onModelChange, contacts, onSelectContact }: AskViewProps) {
  const { chats, activeChatId, pending, errors, status } = useSyncExternalStore(
    chatStore.subscribe,
    chatStore.getState,
  )
  const [draft, setDraft] = useState('')
  const [scope, setScope] = useState('')
  const [scopeChatId, setScopeChatId] = useState<string | null>(activeChatId)
  const threadRef = useRef<HTMLDivElement | null>(null)

  const activeChat = useMemo(
    () => chats.find((chat) => chat.id === activeChatId) ?? null,
    [chats, activeChatId],
  )
  const messages = activeChat?.messages ?? []
  const isPending = activeChatId ? Boolean(pending[activeChatId]) : false
  const error = activeChatId ? errors[activeChatId] ?? '' : ''
  const pendingStatus = activeChatId ? status[activeChatId] ?? '' : ''
  // once the answer starts streaming, the growing message replaces the dots
  const isStreamingAnswer = isPending && messages.some((message) => message.streaming)

  // reset the composer scope when the active chat changes (no effect needed)
  if (activeChatId !== scopeChatId) {
    setScopeChatId(activeChatId)
    setScope(activeChat?.scope ?? '')
  }

  // follow the answer as it streams: track content growth, not just message
  // count, and only auto-scroll when the user is already near the bottom (so
  // scrolling up to reread is never hijacked)
  const lastContentLength = messages.length
    ? messages[messages.length - 1].content.length
    : 0
  useEffect(() => {
    const thread = threadRef.current
    if (!thread) return
    const nearBottom = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 160
    if (nearBottom || !isStreamingAnswer) {
      thread.scrollTo({ top: thread.scrollHeight, behavior: isStreamingAnswer ? 'auto' : 'smooth' })
    }
  }, [messages.length, isPending, lastContentLength, isStreamingAnswer])

  function newChat() {
    chatStore.setActive(null)
    setDraft('')
  }

  function openChat(id: string) {
    chatStore.setActive(id)
    setDraft('')
  }

  function deleteChat(id: string) {
    chatStore.deleteChat(id)
  }

  function submit(nextQuestion = draft) {
    const question = nextQuestion.trim()
    if (!defaults?.messagesPath || !question) return
    if (activeChatId && pending[activeChatId]) return
    setDraft('')
    void chatStore.send(question, scope, {
      messagesPath: defaults.messagesPath,
      question,
      contact: scope,
      model,
      limit: 10,
    })
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
                  {message.role === 'assistant' ? (
                    <AnswerBody text={message.content} />
                  ) : (
                    <p>{message.content}</p>
                  )}
                  {message.streaming ? <span className="stream-caret" aria-hidden /> : null}
                  {message.role === 'assistant' &&
                  message.response?.mode === 'local' &&
                  message.response.citations.length > 0 ? (
                    <span className="chat-mode-note">Keyword match — the AI answer was unavailable.</span>
                  ) : null}
                  {message.role === 'assistant' &&
                  message.response &&
                  message.response.citations.length ? (
                    <details className="chat-refs">
                      <summary>
                        <span className="chat-refs-count">{message.response.citations.length}</span>
                        referenced {message.response.citations.length === 1 ? 'message' : 'messages'}
                      </summary>
                      <div className="chat-refs-list">
                        {message.response.citations.map((citation, idx) => (
                          <button
                            key={`${citation.chatId}-${citation.timestamp}-${idx}`}
                            type="button"
                            className="chat-ref-row"
                            onClick={() => openCitation(citation.chatId)}
                          >
                            <span className="chat-ref-index">{idx + 1}</span>
                            <span className="chat-ref-body">
                              <span className="chat-ref-head">
                                {citation.senderName || contactLabel(citation.displayName, citation.chatId)}
                                {citation.senderName && (citation.chatId || '').startsWith('chat')
                                  ? ` · ${contactLabel(citation.displayName, citation.chatId)}`
                                  : ''}
                                {' · '}
                                {shortDateTime(citation.timestamp)}
                              </span>
                              <span className={`chat-ref-text ${isOutbound(citation) ? 'outbound' : ''}`}>
                                {citation.text || '(empty)'}
                              </span>
                            </span>
                          </button>
                        ))}
                      </div>
                    </details>
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

          {isPending && !isStreamingAnswer ? (
            <article className="chat-message assistant">
              <div className="chat-avatar">R</div>
              <div className="chat-bubble typing">
                <span />
                <span />
                <span />
                {pendingStatus ? <em className="chat-status">{pendingStatus}</em> : null}
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
              disabled={!defaults?.messagesPath || !draft.trim() || isPending}
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
