import { useEffect, useMemo, useRef, type KeyboardEvent } from 'react'
import { contactTitle, messageCount } from '../../lib/conversations'
import { formatNumber } from '../../lib/format'
import { rovingKeyDown } from '../../lib/roving'
import type { Contact } from '../../types'
import { SearchField } from '../ui/SearchField'
import { Truncate } from '../ui/Truncate'

type ConversationListProps = {
  contacts: Contact[]
  selectedId: string | null
  loading: boolean
  query: string
  onQueryChange: (query: string) => void
  onSelect: (contact: Contact) => void
}

const SKELETON_ROWS = ['Name of a person', 'Another name', 'A longer group name here', 'Short name', 'Name', 'One more name']

export function ConversationList({
  contacts,
  selectedId,
  loading,
  query,
  onQueryChange,
  onSelect,
}: ConversationListProps) {
  const list = useRef<HTMLUListElement>(null)
  const named = useMemo(() => contacts.map((contact) => ({ contact, title: contactTitle(contact) })), [contacts])
  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (!needle) return named
    // a name, or part of the number or address the conversation is with
    return named.filter(({ contact, title }) => `${title} ${contact.chat_id}`.toLowerCase().includes(needle))
  }, [named, query])

  // a conversation chosen somewhere else is brought into view here
  useEffect(() => {
    list.current?.querySelector('[aria-current="true"]')?.scrollIntoView({ block: 'nearest' })
  }, [selectedId])

  function handleSearchKey(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      const rows = list.current?.querySelectorAll<HTMLElement>('[data-roving]')
      const current = list.current?.querySelector<HTMLElement>('[data-roving][tabindex="0"]')
      ;(current ?? rows?.[0])?.focus()
    } else if (event.key === 'Enter' && matches[0]) {
      onSelect(matches[0].contact)
    }
  }

  const selectedIsShown = matches.some(({ contact }) => contact.chat_id === selectedId)

  return (
    <aside className="pane appear" aria-label="Conversations">
      <div className="pane-head pane-actions">
        <SearchField value={query} onChange={onQueryChange} label="Search conversations" onKeyDown={handleSearchKey} />
      </div>

      <div className="pane-scroll">
        {loading && !contacts.length ? (
          <ul className="rows" aria-hidden="true">
            {SKELETON_ROWS.map((name) => (
              <li key={name} className="row">
                <span>
                  <span className="skeleton">{name}</span>
                </span>
              </li>
            ))}
          </ul>
        ) : matches.length ? (
          <ul className="rows conversations" role="list" ref={list} onKeyDown={rovingKeyDown}>
            {matches.map(({ contact, title }, index) => {
              const active = contact.chat_id === selectedId
              return (
                <li key={contact.chat_id}>
                  <button
                    type="button"
                    className={active ? 'row is-active' : 'row'}
                    aria-current={active ? 'true' : undefined}
                    data-roving
                    // Tab stops at the chosen conversation, or the first when it is filtered out
                    tabIndex={active || (!selectedIsShown && index === 0) ? 0 : -1}
                    onClick={() => onSelect(contact)}
                  >
                    <span className="row-main">
                      <Truncate className="row-title">{title}</Truncate>
                    </span>
                    <span className="row-end">
                      {formatNumber(messageCount(contact))}
                      <span className="sr-only"> messages</span>
                    </span>
                  </button>
                </li>
              )
            })}
          </ul>
        ) : (
          <p className="pane-note">{query.trim() ? `No conversations match “${query.trim()}”` : 'No conversations yet'}</p>
        )}
      </div>
    </aside>
  )
}
