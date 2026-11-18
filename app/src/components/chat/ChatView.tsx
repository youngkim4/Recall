import { useEffect, useId, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import { bookmarkId } from '../../lib/bookmarks'
import { chatStore, type ChatMessage } from '../../lib/chatStore'
import { contactTitle, conversationName } from '../../lib/conversations'
import { dayOfMessage } from '../../lib/dayParts'
import { scopeChoices } from '../../lib/scope'
import { useMediaQuery } from '../../lib/useMediaQuery'
import { useReasoning } from '../../lib/useReasoning'
import type { Contact, Defaults, SearchResult } from '../../types'
import { ChevronLeftIcon } from '../Icons'
import { DaySheet } from '../ui/DaySheet'
import { ChatCites } from './ChatCites'
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

// a window this wide has room for the cited messages beside the thread
const ROOM_BESIDE = '(min-width: 80rem)'

const citedBy = (message: ChatMessage) => message.response?.citations ?? []

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
  // how hard the model in use is asked to think; the slider in the Quality menu sets it
  const { effort } = useReasoning(model)
  const [scope, setScope] = useState('')
  const [scopeChatId, setScopeChatId] = useState<string | null>(activeChatId)
  const [openedAt, setOpenedAt] = useState(() => Date.now())
  // in a narrow window only one of the two panes shows
  const [pane, setPane] = useState<'list' | 'detail'>('detail')
  // the answer whose cited messages the reader asked for, and how many answers cited anything then
  const [chosen, setChosen] = useState<{ id: string; among: number } | null>(null)
  // the cited message whose day is open
  const [opened, setOpened] = useState<SearchResult | null>(null)
  const citesId = useId()
  const roomBeside = useMediaQuery(ROOM_BESIDE)
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

  // Beside the thread stand the messages of the newest answer that cites any.
  // The reader's choice of another answer holds until a new one arrives.
  const citing = messages.filter((message) => message.role === 'assistant' && citedBy(message).length > 0)
  const asked = chosen && chosen.among === citing.length ? citing.find((message) => message.id === chosen.id) : undefined
  const beside = roomBeside ? (asked ?? citing[citing.length - 1] ?? null) : null
  const openedContact = opened ? contacts.find((contact) => contact.chat_id === opened.chatId) : undefined
  const openedDay = opened ? dayOfMessage(opened.timestamp) : null

  // A cited message opens on its day. One the export gives no time for is in
  // no day, so it opens its conversation.
  function openCited(citation: SearchResult) {
    if (dayOfMessage(citation.timestamp)) {
      setOpened(citation)
      return
    }
    const contact = contacts.find((item) => item.chat_id === citation.chatId)
    onSelectContact(contact ?? { chat_id: citation.chatId, display_name: citation.displayName })
  }

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
      reasoningEffort: effort,
      limit: 10,
    })
  }

  return (
    <>
      <div className="split chat" data-pane={pane} data-cites={beside ? '' : undefined}>
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
                beside={beside ? { paneId: citesId, answerId: beside.id } : null}
                onShowCited={(id) => setChosen({ id, among: citing.length })}
                openKey={opened ? bookmarkId(opened) : ''}
                onOpenCitation={openCited}
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
              pending={isPending}
              scope={scope}
              scopeOptions={scopeOptions}
              model={model}
              models={defaults?.models ?? []}
              onDraftChange={setDraft}
              onScopeChange={setScope}
              onModelChange={onModelChange}
              onSubmit={() => submit()}
              onStop={() => {
                if (activeChatId) chatStore.stop(activeChatId)
              }}
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

        {beside ? (
          <>
            <div className="divider" />
            <ChatCites
              id={citesId}
              answerId={beside.id}
              citations={citedBy(beside)}
              openKey={opened ? bookmarkId(opened) : ''}
              onOpen={openCited}
            />
          </>
        ) : null}
      </div>

      {/* a cited message opens on its day, over the chat; the sheet is not one of the panes */}
      {opened && openedDay ? (
        <DaySheet
          key={bookmarkId(opened)}
          messagesPath={defaults?.messagesPath ?? ''}
          chatId={opened.chatId}
          title={openedContact ? contactTitle(openedContact) : conversationName(opened.displayName, opened.chatId)}
          day={openedDay}
          at={opened.timestamp}
          pickedId={opened.messageId}
          onOpenConversation={() =>
            onSelectContact(openedContact ?? { chat_id: opened.chatId, display_name: opened.displayName })
          }
          onClose={() => setOpened(null)}
        />
      ) : null}
    </>
  )
}
