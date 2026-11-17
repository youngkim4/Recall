import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { recallApi } from '../../lib/api'
import { bookmarkId, readBookmarks, toggleBookmark } from '../../lib/bookmarks'
import { formatNumber, plural } from '../../lib/format'
import { rovingKeyDown } from '../../lib/roving'
import { scopeChoices } from '../../lib/scope'
import type { Contact, Defaults, SearchResult } from '../../types'
import { BookmarkIcon } from '../Icons'
import { ListPicker } from '../ui/ListPicker'
import { MessageHit } from '../ui/MessageHit'
import { SearchField } from '../ui/SearchField'

type SearchViewProps = {
  defaults: Defaults | null
  contacts: Contact[]
  onSelectContact: (contact: Contact) => void
}

// the newest matches are listed; the count says how many there are in all
const SHOWN = 80
// typing pauses this long before the archive is searched
const WAIT_MS = 250
const FEWEST_LETTERS = 2

// what was found, and the search it answers
type Found = { key: string; results: SearchResult[]; total: number }

// Exact words, found as they are typed: each match as the message it was,
// with the words picked out. With nothing typed, the messages saved from
// earlier searches.
export function SearchView({ defaults, contacts, onSelectContact }: SearchViewProps) {
  const messagesPath = defaults?.messagesPath ?? ''
  const list = useRef<HTMLUListElement>(null)
  const [query, setQuery] = useState('')
  const [scope, setScope] = useState('')
  const [found, setFound] = useState<Found | null>(null)
  const [failedFor, setFailedFor] = useState('')
  const [attempt, setAttempt] = useState(0)
  const [saved, setSaved] = useState<SearchResult[]>(() => readBookmarks())

  const term = query.trim()
  const searching = term.length >= FEWEST_LETTERS
  const key = `${term}\n${scope}`
  const scopes = useMemo(() => scopeChoices(contacts), [contacts])
  const savedIds = useMemo(() => new Set(saved.map(bookmarkId)), [saved])

  useEffect(() => {
    if (!messagesPath || !searching) return
    let cancelled = false
    const timer = window.setTimeout(() => {
      recallApi
        .search({ messagesPath, query: term, contact: scope, limit: SHOWN })
        .then((response) => {
          if (cancelled) return
          setFound({ key, results: response.results ?? [], total: response.total ?? response.count })
          setFailedFor('')
        })
        .catch(() => {
          if (!cancelled) setFailedFor(key)
        })
    }, WAIT_MS)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [messagesPath, searching, term, scope, key, attempt])

  // while the next answer is on its way, the last one stays up
  const settled = found?.key === key
  const failed = failedFor === key
  const shown = searching ? (found?.results ?? []) : saved

  function open(message: SearchResult) {
    onSelectContact(
      contacts.find((contact) => contact.chat_id === message.chatId) ?? {
        chat_id: message.chatId,
        display_name: message.displayName,
      },
    )
  }

  function handleSearchKey(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key !== 'ArrowDown') return
    event.preventDefault()
    list.current?.querySelector<HTMLElement>('[data-roving]')?.focus()
  }

  function handleListKey(event: KeyboardEvent<HTMLUListElement>) {
    // S saves the message the keyboard is on, or takes it back out
    if (event.key.toLowerCase() === 's' && !event.metaKey && !event.ctrlKey && !event.altKey) {
      const rows = [...event.currentTarget.querySelectorAll<HTMLElement>('[data-roving]')]
      const at = rows.indexOf(document.activeElement as HTMLElement)
      const message = shown[at]
      if (message) {
        event.preventDefault()
        // among the saved, the row goes away: the next one takes the focus, or the one before
        if (!searching) (rows[at + 1] ?? rows[at - 1])?.focus()
        setSaved(toggleBookmark(saved, message))
      }
      return
    }
    rovingKeyDown(event)
  }

  const status = !searching
    ? saved.length
      ? `${formatNumber(saved.length)} saved`
      : ''
    : failed
      ? ''
      : !settled
        ? 'Searching'
        : found.total === 0
          ? `No messages match “${term}”`
          : found.total > found.results.length
            ? `${plural(found.total, 'message')}, the newest ${formatNumber(found.results.length)} shown`
            : plural(found.total, 'message')

  return (
    <div className="view">
      <div className="page arrive-each">
        <div className="search-bar">
          <SearchField
            value={query}
            onChange={setQuery}
            label="Search messages"
            placeholder="Search messages"
            large
            autoFocus
            disabled={!messagesPath}
            onKeyDown={handleSearchKey}
          />
          <ListPicker
            label="Conversations to search"
            value={scope}
            options={scopes}
            onChange={setScope}
            searchPlaceholder="Search conversations"
            emptyText="No conversations match"
          />
        </div>

        {failed ? (
          <p className="search-status" role="alert">
            The archive could not be searched.{' '}
            <button type="button" className="link in-text" onClick={() => setAttempt((current) => current + 1)}>
              Try again
            </button>
          </p>
        ) : (
          <p className="search-status" role="status">
            {status}
          </p>
        )}

        {shown.length ? (
          <ul
            ref={list}
            className="hits"
            role="list"
            aria-label={searching ? 'Matches' : 'Saved messages'}
            onKeyDown={handleListKey}
          >
            {shown.map((message, index) => {
              const id = bookmarkId(message)
              const isSaved = savedIds.has(id)
              return (
                <li key={id}>
                  <MessageHit
                    message={message}
                    term={searching ? term : ''}
                    onOpen={() => open(message)}
                    roving
                    shortcut="S"
                    tabIndex={index === 0 ? 0 : -1}
                    action={
                      <button
                        type="button"
                        className="icon-btn sm"
                        aria-pressed={isSaved}
                        aria-label={isSaved ? 'Remove from saved' : 'Save message'}
                        aria-keyshortcuts="S"
                        title={isSaved ? 'Remove from saved (S)' : 'Save (S)'}
                        // reached with S from the row, so Tab passes over it
                        tabIndex={-1}
                        onClick={() => setSaved(toggleBookmark(saved, message))}
                      >
                        {/* the mark swaps with a small pop when it changes */}
                        <span key={String(isSaved)} className="icon-swap">
                          <BookmarkIcon filled={isSaved} />
                        </span>
                      </button>
                    }
                  />
                </li>
              )
            })}
          </ul>
        ) : null}
      </div>
    </div>
  )
}
