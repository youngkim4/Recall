import { useMemo, useState } from 'react'
import { SearchIcon } from './Icons'
import { recallApi } from '../lib/api'
import { contactTitle, formatNumber, isOutbound, prettyContactId, shortDateTime } from '../lib/format'
import type { Contact, Defaults, SearchResult } from '../types'

const BOOKMARKS_KEY = 'recall.bookmarks.v1'

type ExplorerViewProps = {
  defaults: Defaults | null
  contacts: Contact[]
  selectedContact: Contact | null
  onSelectContact: (contact: Contact) => void
}

export function ExplorerView({
  defaults,
  contacts,
  selectedContact,
  onSelectContact,
}: ExplorerViewProps) {
  const [query, setQuery] = useState('')
  const [scope, setScope] = useState(selectedContact?.chat_id || '')
  const [results, setResults] = useState<SearchResult[]>([])
  const [searching, setSearching] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)
  const [error, setError] = useState('')
  const [bookmarks, setBookmarks] = useState<SearchResult[]>(() => readBookmarks())

  const scopedContact = useMemo(
    () => contacts.find((contact) => contact.chat_id === scope) || null,
    [contacts, scope],
  )

  async function runSearch() {
    if (!defaults?.messagesPath) return
    setSearching(true)
    setHasSearched(true)
    setError('')
    try {
      const response = await recallApi.search({
        messagesPath: defaults.messagesPath,
        query,
        contact: scope,
        limit: 80,
      })
      setResults(response.results || [])
    } catch (apiError) {
      setResults([])
      setError(apiError instanceof Error ? apiError.message : 'Unable to search messages.')
    } finally {
      setSearching(false)
    }
  }

  function saveBookmark(result: SearchResult) {
    const id = bookmarkId(result)
    const next = [result, ...bookmarks.filter((item) => bookmarkId(item) !== id)].slice(0, 30)
    setBookmarks(next)
    storeBookmarks(next)
  }

  function openResult(result: SearchResult) {
    const contact = contacts.find((item) => item.chat_id === result.chatId)
    if (contact) onSelectContact(contact)
  }

  const showResults = searching || hasSearched || results.length > 0
  const showBookmarks = bookmarks.length > 0

  return (
    <section className="explorer-view" aria-label="Explore messages">
      <div className="explorer-header">
        <div>
          <span className="eyebrow">Search</span>
          <h2>Search the archive</h2>
          <p>Find exact messages and save moments without generating a report.</p>
        </div>
      </div>

      <div className="explorer-tools">
        <label className="search-field no-margin">
          <SearchIcon className="field-icon" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') void runSearch()
            }}
            placeholder="Search exact text"
            aria-label="Search message text"
          />
        </label>
        <label>
          <span>Scope</span>
          <select value={scope} onChange={(event) => setScope(event.target.value)}>
            <option value="">All conversations</option>
            {contacts.map((contact) => (
              <option key={contact.chat_id} value={contact.chat_id}>
                {contactTitle(contact)}
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="button primary" disabled={!defaults?.messagesPath || searching} onClick={() => void runSearch()}>
          {searching ? 'Searching...' : 'Search'}
        </button>
      </div>

      {error ? <div className="inline-error">{error}</div> : null}

      {showResults || showBookmarks ? (
        <div className={`explorer-grid ${showResults && showBookmarks ? '' : 'single'}`}>
          {showResults ? (
            <section className="results-panel">
              <div className="block-heading">
                <div>
                  <h2>Results</h2>
                  <p>
                    {searching
                      ? 'Searching...'
                      : results.length
                        ? `${formatNumber(results.length)} messages${scopedContact ? ` in ${contactTitle(scopedContact)}` : ''}.`
                        : 'No matches found.'}
                  </p>
                </div>
              </div>
              {results.length ? (
                <MessageResults results={results} onOpen={openResult} onBookmark={saveBookmark} />
              ) : null}
            </section>
          ) : null}

          {showBookmarks ? (
            <aside className="bookmark-panel">
              <div className="block-heading">
                <div>
                  <h2>Saved Moments</h2>
                  <p>{formatNumber(bookmarks.length)} local bookmarks.</p>
                </div>
              </div>
              <MessageResults compact results={bookmarks} onOpen={openResult} onBookmark={saveBookmark} />
            </aside>
          ) : null}
        </div>
      ) : null}
    </section>
  )
}

function MessageResults({
  results,
  compact,
  onOpen,
  onBookmark,
}: {
  results: SearchResult[]
  compact?: boolean
  onOpen: (result: SearchResult) => void
  onBookmark: (result: SearchResult) => void
}) {
  if (!results.length) return <div className="empty-inline">No messages to show.</div>
  return (
    <div className={`search-results ${compact ? 'compact' : ''}`}>
      {results.map((result) => (
        <article key={bookmarkId(result)} className="search-result">
          <button type="button" className="search-result-main" onClick={() => onOpen(result)}>
            <span>
              {result.displayName || prettyContactId(result.chatId)} / {shortDateTime(result.timestamp)}
            </span>
            <p className={isOutbound(result) ? 'outbound' : ''}>{result.text || '(empty)'}</p>
          </button>
          <button type="button" className="mini-button" onClick={() => onBookmark(result)}>
            Save
          </button>
        </article>
      ))}
    </div>
  )
}

function bookmarkId(result: SearchResult) {
  return `${result.chatId}-${result.timestamp}-${result.messageId || result.text.slice(0, 24)}`
}

function readBookmarks() {
  try {
    const stored = window.localStorage.getItem(BOOKMARKS_KEY)
    return stored ? (JSON.parse(stored) as SearchResult[]) : []
  } catch {
    return []
  }
}

function storeBookmarks(bookmarks: SearchResult[]) {
  try {
    window.localStorage.setItem(BOOKMARKS_KEY, JSON.stringify(bookmarks))
  } catch {
    // Bookmarks are a local convenience only.
  }
}
