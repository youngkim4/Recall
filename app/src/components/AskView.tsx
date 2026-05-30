import { useEffect, useMemo, useRef, useState } from 'react'
import { SparkIcon } from './Icons'
import { recallApi } from '../lib/api'
import { contactTitle, isOutbound, shortDateTime } from '../lib/format'
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
  scopeLabel?: string
  response?: AskResponse
}

const starterPrompts = [
  'What changed between us over time?',
  'Find moments where trust came up.',
  'What did we talk about around travel?',
  'Summarize the most emotional messages.',
]

export function AskView({ defaults, model, onModelChange, contacts, onSelectContact }: AskViewProps) {
  const [draft, setDraft] = useState('')
  const [scope, setScope] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const threadRef = useRef<HTMLDivElement | null>(null)
  const messageId = useRef(0)
  const effectiveScope = scope

  const scopedContact = useMemo(
    () => contacts.find((contact) => contact.chat_id === effectiveScope) || null,
    [contacts, effectiveScope],
  )
  const scopeLabel = scopedContact ? contactTitle(scopedContact) : 'All conversations'

  useEffect(() => {
    threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, loading])

  async function submit(nextQuestion = draft) {
    const question = nextQuestion.trim()
    if (!defaults?.messagesPath || !question || loading) return

    const userMessage: ChatMessage = {
      id: `user-${(messageId.current += 1)}`,
      role: 'user',
      content: question,
      scopeLabel,
    }

    setMessages((current) => [...current, userMessage])
    setDraft('')
    setLoading(true)
    setError('')

    try {
      const payload = await recallApi.ask({
        messagesPath: defaults.messagesPath,
        question,
        contact: effectiveScope,
        model,
        limit: 10,
      })
      setMessages((current) => [
        ...current,
        {
          id: `assistant-${(messageId.current += 1)}`,
          role: 'assistant',
          content: payload.answer,
          response: payload,
          scopeLabel,
        },
      ])
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
    <section className={`ask-view chat-view ${messages.length ? '' : 'is-empty'}`} aria-label="Recall">
      <div className="chat-thread" ref={threadRef}>
        {messages.length ? (
          messages.map((message) => (
            <article key={message.id} className={`chat-message ${message.role}`}>
              <div className="chat-avatar">{message.role === 'user' ? 'Y' : 'R'}</div>
              <div className="chat-bubble">
                <div className="chat-meta">
                  <strong>{message.role === 'user' ? 'You' : 'Recall'}</strong>
                  <span>{message.scopeLabel}</span>
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
                              {citation.displayName || citation.chatId} / {shortDateTime(citation.timestamp)}
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
            <div className="chat-mark">
              <SparkIcon className="button-icon" />
            </div>
            <span className="eyebrow">Recall</span>
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
            <select value={effectiveScope} onChange={(event) => setScope(event.target.value)} aria-label="Conversation scope">
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
  )
}
