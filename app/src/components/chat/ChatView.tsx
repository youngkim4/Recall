import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import { chatStore } from '../../lib/chatStore'
import { scopeChoices } from '../../lib/scope'
import type { Contact, Defaults } from '../../types'
import { ChevronLeftIcon } from '../Icons'
import { ChatList } from './ChatList'
import { ChatThread } from './ChatThread'
import { Composer } from './Composer'

type ChatViewProps = {
  defaults: Defaults | null
  model: string
  contacts: Contact[]
  onModelChange: (model: string) => void
  onSelectContact: (contact: Contact) => void
  // opens the guide that builds the archive
  onOpenSetup: () => void
}

const starterPrompts = [
  'What changed between us over time?',
  'Find moments where trust came up.',
  'What did we talk about around travel?',
  'Summarize the most emotional messages.',
]

export function ChatView({
  defaults,
  model,
  contacts,
  onModelChange,
  onSelectContact,
  onOpenSetup,
}: ChatViewProps) {
  const { chats, activeChatId, pending, errors, status } = useSyncExternalStore(
    chatStore.subscribe,
    chatStore.getState,
  )
  const [draft, setDraft] = useState('')
  const [scope, setScope] = useState('')
  const [scopeChatId, setScopeChatId] = useState<string | null>(activeChatId)
  const [openedAt, setOpenedAt] = useState(() => Date.now())
  // in a narrow window only one of the two panes shows
  const [pane, setPane] = useState<'list' | 'detail'>('detail')
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const inputRef = useRef<HTMLTextAreaElement | null>(null)

  const activeChat = useMemo(
    () => chats.find((chat) => chat.id === activeChatId) ?? null,
    [chats, activeChatId],
  )
  const messages = activeChat?.messages ?? []
  const isPending = activeChatId ? Boolean(pending[activeChatId]) : false
  const error = activeChatId ? errors[activeChatId] ?? '' : ''
  const pendingStatus = activeChatId ? status[activeChatId] ?? '' : ''
  // once the answer starts streaming, the growing text replaces the typing dots
  const isStreaming = isPending && messages.some((message) => message.streaming)
  const hasArchive = Boolean(defaults?.messagesPath && defaults.hasMessages)
  const isEmpty = messages.length === 0

  // the composer's scope follows the chat that is open
  if (activeChatId !== scopeChatId) {
    setScopeChatId(activeChatId)
    setScope(activeChat?.scope ?? '')
  }

  // Follow the answer as it streams: track content growth, not just message
  // count, and only when the reader is already near the bottom, so scrolling
  // up to reread is never pulled back down.
  const lastLength = messages.length ? messages[messages.length - 1].content.length : 0
  useEffect(() => {
    const scroller = scrollRef.current
    if (!scroller) return
    const nearBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 160
    if (nearBottom || !isStreaming) {
      scroller.scrollTo({ top: scroller.scrollHeight, behavior: isStreaming ? 'auto' : 'smooth' })
    }
  }, [messages.length, isPending, lastLength, isStreaming])

  const scopeOptions = useMemo(() => scopeChoices(contacts), [contacts])

  // the composer is the one thing to type into here, so the caret starts there
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  function newChat() {
    chatStore.setActive(null)
    setDraft('')
    setOpenedAt(Date.now())
    setPane('detail')
    inputRef.current?.focus()
  }

  function openChat(id: string) {
    chatStore.setActive(id)
    setDraft('')
    setOpenedAt(Date.now())
    setPane('detail')
    inputRef.current?.focus()
  }

  function submit(nextQuestion = draft) {
    const question = nextQuestion.trim()
    if (!defaults?.messagesPath || !question || isPending) return
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
    <div className="split chat" data-pane={pane}>
      <ChatList chats={chats} activeChatId={activeChatId} onOpen={openChat} onNew={newChat} />
      <div className="divider" />

      <section className={`pane chat-pane${isEmpty ? ' is-empty' : ''}`} aria-label="Chat">
        <button type="button" className="pane-back btn quiet" onClick={() => setPane('list')}>
          <ChevronLeftIcon />
          Chats
        </button>
        <div className="chat-scroll" ref={scrollRef}>
          {isEmpty ? (
            <h2 key={activeChatId} className="chat-start arrive">
              What do you want to remember?
            </h2>
          ) : (
            <ChatThread
              key={activeChatId}
              messages={messages}
              openedAt={openedAt}
              waiting={isPending && !isStreaming}
              waitingStatus={pendingStatus}
              onOpenCitation={openCitation}
            />
          )}
        </div>

        <div className="chat-dock">
          {error ? (
            <p className="notice bad" role="alert">
              {error}
            </p>
          ) : null}
          <Composer
            inputRef={inputRef}
            draft={draft}
            canSend={hasArchive && Boolean(draft.trim()) && !isPending}
            disabled={Boolean(defaults) && !hasArchive}
            scope={scope}
            scopeOptions={scopeOptions}
            model={model}
            models={defaults?.models ?? []}
            onDraftChange={setDraft}
            onScopeChange={setScope}
            onModelChange={onModelChange}
            onSubmit={() => submit()}
          />
        </div>

        {isEmpty && defaults && !hasArchive ? (
          <div className="chat-setup">
            <p>No messages imported yet</p>
            <button type="button" className="btn" onClick={onOpenSetup}>
              Build your archive
            </button>
          </div>
        ) : isEmpty ? (
          <ul className="rows prompts" role="list" aria-label="Example questions">
            {starterPrompts.map((prompt) => (
              <li key={prompt}>
                <button type="button" className="row" onClick={() => submit(prompt)}>
                  {prompt}
                </button>
              </li>
            ))}
          </ul>
        ) : null}
      </section>
    </div>
  )
}
