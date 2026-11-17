import { useEffect, useId, useRef, useState } from 'react'
import { CloseIcon, PlusIcon } from '../Icons'
import { chatStore, type SavedChat } from '../../lib/chatStore'
import { relativeTime } from '../../lib/format'
import { Truncate } from '../ui/Truncate'

// how long a deleted chat can be brought back
const UNDO_MS = 8000
// the bar's fade out, with a little room; matches .undo-bar.is-leaving
const LEAVE_MS = 250

type ChatListProps = {
  chats: SavedChat[]
  activeChatId: string | null
  onOpen: (id: string) => void
  onNew: () => void
}

export function ChatList({ chats, activeChatId, onOpen, onNew }: ChatListProps) {
  const [removed, setRemoved] = useState<{ chat: SavedChat; index: number } | null>(null)
  // the bar leaves with a short fade; it is taken away when that has finished
  const [leaving, setLeaving] = useState(false)
  const newChat = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (!removed) return
    const timer = window.setTimeout(() => setLeaving(true), UNDO_MS)
    return () => window.clearTimeout(timer)
  }, [removed])

  useEffect(() => {
    if (!leaving) return
    const timer = window.setTimeout(() => {
      setRemoved(null)
      setLeaving(false)
    }, LEAVE_MS)
    return () => window.clearTimeout(timer)
  }, [leaving])

  function remove(chat: SavedChat, index: number, row: HTMLElement | null) {
    // the row goes away, and the focus would go with it: the next chat takes it, or the one before
    if (row?.contains(document.activeElement)) {
      const neighbor = row.nextElementSibling ?? row.previousElementSibling
      const target = neighbor?.querySelector<HTMLElement>('.chat-open') ?? newChat.current
      target?.focus()
    }
    chatStore.deleteChat(chat.id)
    setLeaving(false)
    setRemoved({ chat, index })
  }

  function undo() {
    if (!removed) return
    chatStore.restoreChat(removed.chat, removed.index)
    setLeaving(true)
  }

  return (
    <aside className="pane chat-list appear" aria-label="Saved chats">
      <div className="pane-head pane-actions">
        <button ref={newChat} type="button" className="row" onClick={onNew}>
          <PlusIcon />
          New chat
        </button>
      </div>

      <div className="pane-scroll">
        {chats.length ? (
          <ul className="rows" role="list">
            {chats.map((chat, index) => (
              <ChatRow
                key={chat.id}
                chat={chat}
                active={chat.id === activeChatId}
                onOpen={() => onOpen(chat.id)}
                onRemove={(row) => remove(chat, index, row)}
              />
            ))}
          </ul>
        ) : (
          <p className="pane-note">No chats yet</p>
        )}
      </div>

      {removed ? (
        <div
          // a second deletion gets a bar of its own, with its own eight seconds
          key={removed.chat.id}
          className={leaving ? 'undo-bar is-leaving' : 'undo-bar'}
          role="status"
        >
          <span>Chat deleted</span>
          <button type="button" className="link" disabled={leaving} onClick={undo}>
            Undo
          </button>
        </div>
      ) : null}
    </aside>
  )
}

type ChatRowProps = {
  chat: SavedChat
  active: boolean
  onOpen: () => void
  // with the row that is going, so the focus can be moved off it first
  onRemove: (row: HTMLElement | null) => void
}

function ChatRow({ chat, active, onOpen, onRemove }: ChatRowProps) {
  const timeId = useId()
  const title = chat.title || 'New chat'

  return (
    <li className={`row has-action chat-row${active ? ' is-active' : ''}`}>
      {/* the name is the title alone; the time is read as its description */}
      <button
        type="button"
        className="chat-open"
        aria-current={active ? 'true' : undefined}
        aria-describedby={timeId}
        onClick={onOpen}
      >
        <Truncate className="row-title">{title}</Truncate>
      </button>
      <span className="chat-trail">
        <span className="row-end" id={timeId}>
          {relativeTime(chat.updatedAt)}
        </span>
        <button
          type="button"
          className="icon-btn sm chat-delete"
          aria-label={`Delete chat: ${title}`}
          title="Delete chat"
          onClick={(event) => onRemove(event.currentTarget.closest('li'))}
        >
          <CloseIcon />
        </button>
      </span>
    </li>
  )
}
