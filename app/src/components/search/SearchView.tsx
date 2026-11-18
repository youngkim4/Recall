import { useEffect, useMemo, useRef, useState, type FocusEvent, type KeyboardEvent } from 'react'
import { recallApi } from '../../lib/api'
import { bookmarkId, readBookmarks, toggleBookmark } from '../../lib/bookmarks'
import { contactTitle, conversationName } from '../../lib/conversations'
import { dayOfMessage } from '../../lib/dayParts'
import { formatNumber, plural } from '../../lib/format'
import { rovingKeyDown } from '../../lib/roving'
import { loadPatterns } from '../../lib/overviewLoads'
import { scopeChoices } from '../../lib/scope'
import { NO_FILTERS, filterParams, filtersKey, filtersSet, shownWord, type SearchFilters } from '../../lib/searchFilters'
import { keepFound, type Found } from '../../lib/searchFound'
import { useLoad } from '../../lib/useLoad'
import type { Contact, Defaults, SearchResult } from '../../types'
import { BookmarkIcon, ChevronLeftIcon } from '../Icons'
import { DayReader } from '../ui/DayReader'
import { MessageHit } from '../ui/MessageHit'
import { SearchFilterBar, SearchSort } from './SearchFilterBar'
import { SearchField } from '../ui/SearchField'
import { Truncate } from '../ui/Truncate'

type SearchViewProps = {
  defaults: Defaults | null
  contacts: Contact[]
  onSelectContact: (contact: Contact) => void
}

// the newest matches are listed; the count says how many there are in all
const SHOWN = 80
// typing pauses this long before the archive is searched, the last word as the start of a word
const WAIT_MS = 250
// and stops this long before it is searched again for whole words
const WHOLE_MS = 900
const FEWEST_LETTERS = 2

// What tells one row of the list from another. A saved message is known by
// its conversation, time and words (bookmarkId), and two messages can share
// all three, so a row goes by the id of its message when it has one.
const rowId = (message: SearchResult) => (message.messageId ? `id:${message.messageId}` : bookmarkId(message))

// Exact words, found as they are typed: each match as the message it was,
// with the words picked out, and beside the list the day around the match
// that is chosen. With nothing typed, the messages saved from earlier
// searches. In a narrow window the list and the day take turns.
export function SearchView({ defaults, contacts, onSelectContact }: SearchViewProps) {
  const messagesPath = defaults?.messagesPath ?? ''
  const list = useRef<HTMLUListElement>(null)
  const [query, setQuery] = useState('')
  const [filters, setFilters] = useState<SearchFilters>(NO_FILTERS)
  // the days of the archive, for the calendar of the date filter (kept from the Overview)
  const { load: patterns } = useLoad(messagesPath, loadPatterns)
  const [found, setFound] = useState<Found | null>(null)
  const [failedFor, setFailedFor] = useState('')
  const [attempt, setAttempt] = useState(0)
  const [saved, setSaved] = useState<SearchResult[]>(() => readBookmarks())
  // the message whose day is open beside the list
  const [pickedId, setPickedId] = useState('')
  const [pane, setPane] = useState<'list' | 'detail'>('list')

  const typed = query.trim()
  const hasWords = typed.length >= FEWEST_LETTERS
  const term = hasWords ? typed : ''
  // words, or a filter with no words: then the filter lists the messages it lets through
  const narrowed = filtersSet(filters)
  const searching = hasWords || narrowed
  const key = `${term}\n${filtersKey(filters)}`
  const scopes = useMemo(() => scopeChoices(contacts), [contacts])
  const savedIds = useMemo(() => new Set(saved.map(bookmarkId)), [saved])

  // Two asks for each search: soon after a key, with the last word as the
  // start of a word, and when typing has stopped, for whole words, so "hi"
  // settles on the word "hi" and not "hit" (keepFound says which stays).
  // A search of filters alone has no words to read either way, and asks once.
  useEffect(() => {
    if (!messagesPath || !searching) return
    let cancelled = false
    const ask = (whole: boolean) =>
      recallApi
        .search({
          messagesPath,
          query: term,
          limit: SHOWN,
          prefix: term ? !whole : undefined,
          filters: filterParams(filters, Boolean(term)),
        })
        .then((response) => {
          if (cancelled) return
          const next = { key, whole, results: response.results ?? [], total: response.total ?? response.count }
          setFound((shown) => keepFound(shown, next))
          setFailedFor('')
        })
        .catch(() => {
          if (!cancelled) setFailedFor(key)
        })
    const soon = window.setTimeout(() => void ask(!term), WAIT_MS)
    const stopped = term ? window.setTimeout(() => void ask(true), WHOLE_MS) : 0
    return () => {
      cancelled = true
      window.clearTimeout(soon)
      window.clearTimeout(stopped)
    }
  }, [messagesPath, searching, term, filters, key, attempt])

  // while the next answer is on its way, the last one stays up
  const settled = found?.key === key
  const failed = failedFor === key
  const shown = searching ? (found?.results ?? []) : saved
  // a list that no longer holds the chosen message shows the day of its first
  const current = shown.find((message) => rowId(message) === pickedId) ?? shown[0] ?? null
  const currentId = current ? rowId(current) : ''
  const currentContact = current ? contacts.find((contact) => contact.chat_id === current.chatId) : undefined
  // a message the export gives no time for is in no day
  const currentDay = current ? dayOfMessage(current.timestamp) : null

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

  // the day beside the list follows the row the keyboard is on
  function handleListFocus(event: FocusEvent<HTMLUListElement>) {
    const rows = [...event.currentTarget.querySelectorAll<HTMLElement>('[data-roving]')]
    const message = shown[rows.indexOf(event.target as HTMLElement)]
    if (message) setPickedId(rowId(message))
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
          ? term
            ? `No messages match “${term}”`
            : 'No messages match these filters'
          : found.total > found.results.length
            ? `${plural(found.total, 'message')}, the ${shownWord(filters, Boolean(term))} ${formatNumber(found.results.length)} shown`
            : plural(found.total, 'message')

  return (
    <div className="split search-split" data-pane={current ? pane : 'list'}>
      <section className="pane appear" aria-label="Search">
        <div className="search-head">
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
          </div>
          <SearchFilterBar
            filters={filters}
            onChange={setFilters}
            scopes={scopes}
            daily={patterns.state === 'ready' ? patterns.value.daily : null}
          />

          {failed ? (
            <p className="search-status" role="alert">
              The archive could not be searched.{' '}
              <button type="button" className="link in-text" onClick={() => setAttempt((count) => count + 1)}>
                Try again
              </button>
            </p>
          ) : (
            <div className="search-status-row">
              <p className="search-status" role="status">
                {status}
              </p>
              {searching ? (
                <SearchSort
                  order={filters.order}
                  onChange={(order) => setFilters({ ...filters, order })}
                  hasWords={hasWords}
                />
              ) : null}
            </div>
          )}
        </div>

        <div className="pane-scroll">
          {shown.length ? (
            <ul
              ref={list}
              className="hits"
              role="list"
              aria-label={searching ? 'Matches' : 'Saved messages'}
              onKeyDown={handleListKey}
              onFocus={handleListFocus}
            >
              {shown.map((message, index) => {
                // the row by its own id: two messages can share a second and their words
                const id = rowId(message)
                const isSaved = savedIds.has(bookmarkId(message))
                return (
                  <li key={`${id}\n${index}`}>
                    <MessageHit
                      message={message}
                      term={term}
                      picked={id === currentId}
                      onOpen={() => {
                        setPickedId(id)
                        setPane('detail')
                      }}
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
      </section>

      <div className="divider" />

      <section className="pane search-day" aria-label="The day around the message">
        {current ? (
          <>
            <header className="search-day-head">
              <button type="button" className="pane-back btn quiet" onClick={() => setPane('list')}>
                <ChevronLeftIcon />
                {searching ? 'Matches' : 'Saved'}
              </button>
              <h2>
                <Truncate className="dayview-title">
                  {currentContact ? contactTitle(currentContact) : conversationName(current.displayName, current.chatId)}
                </Truncate>
              </h2>
              <button type="button" className="link" onClick={() => open(current)}>
                Open conversation
              </button>
            </header>
            {currentDay ? (
              <DayReader
                key={currentId}
                messagesPath={messagesPath}
                chatId={current.chatId}
                day={currentDay}
                at={current.timestamp}
                pickedId={current.messageId}
                label="Messages of this day"
              />
            ) : (
              <div className="pane-empty">
                <p>This message has no date</p>
              </div>
            )}
          </>
        ) : (
          <div className="pane-empty">
            <p>No message selected</p>
          </div>
        )}
      </section>
    </div>
  )
}
