import { useMemo } from 'react'
import { SearchIcon } from './Icons'
import type { Contact } from '../types'
import { contactTitle, formatNumber, initials, shortDate } from '../lib/format'

type ConversationListProps = {
  contacts: Contact[]
  selectedContact: Contact | null
  loading: boolean
  query: string
  contactNameCount: number
  onQueryChange: (query: string) => void
  onSelect: (contact: Contact) => void
}

export function ConversationList({
  contacts,
  selectedContact,
  loading,
  query,
  contactNameCount,
  onQueryChange,
  onSelect,
}: ConversationListProps) {
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (!needle) return contacts
    return contacts.filter((contact) =>
      [contact.chat_id, contact.display_name, contact.displayName]
        .filter(Boolean)
        .join(' ')
        .toLowerCase()
        .includes(needle),
    )
  }, [contacts, query])

  return (
    <section className="conversation-pane" aria-label="Conversations">
      <div className="pane-heading compact">
        <div>
          <h2>Conversations</h2>
          <p>
            {loading
              ? 'Loading...'
              : `${formatNumber(filtered.length)} of ${formatNumber(contacts.length)} conversations`}
          </p>
        </div>
        <span className="named-pill">{formatNumber(contactNameCount)} named</span>
      </div>

      <label className="search-field">
        <SearchIcon className="field-icon" />
        <input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="Search"
          aria-label="Search conversations"
        />
      </label>

      <div className="conversation-list">
        {filtered.map((contact) => {
          const active = selectedContact?.chat_id === contact.chat_id
          const isGroup = Number(contact.is_group) === 1
          const name = contactTitle(contact)
          const meta = [
            isGroup ? 'Group' : '',
            `${shortDate(contact.first_msg)} to ${shortDate(contact.last_msg)}`,
          ]
            .filter(Boolean)
            .join(' / ')

          return (
            <button
              key={contact.chat_id}
              type="button"
              className={`conversation-row ${active ? 'active' : ''}`}
              onClick={() => onSelect(contact)}
              aria-pressed={active}
            >
              <span className="avatar">{initials(name)}</span>
              <span className="conversation-main">
                <strong>{name}</strong>
                <span>{meta}</span>
              </span>
              <span className="count-pill">{formatNumber(contact.message_count || 0)}</span>
            </button>
          )
        })}

        {!loading && filtered.length === 0 ? (
          <div className="empty-state">
            <strong>No conversations found.</strong>
            <span>Try a different name or handle.</span>
          </div>
        ) : null}
      </div>
    </section>
  )
}
