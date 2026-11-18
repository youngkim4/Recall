import type { ReactNode } from 'react'
import { isOutbound, messageSource } from '../../lib/conversations'
import { formatDateTime } from '../../lib/format'
import { splitByRanges, splitByTerm } from '../../lib/highlight'
import type { SearchResult } from '../../types'

// What to pick out: the ranges the server matched, when it sent them, so
// "hi" does not light up inside "this"; from an older server, the term
// wherever it stands. A message with no text has nothing to pick out.
function marked(message: SearchResult, words: string, term: string) {
  if (!message.text || !term) return splitByTerm(words, '')
  if (Array.isArray(message.ranges)) return splitByRanges(words, message.ranges)
  return splitByTerm(words, term)
}

type MessageHitProps = {
  message: SearchResult
  // its place in a numbered list, when it is in one
  index?: number
  // words to pick out in the text
  term?: string
  // the row whose message is open beside the list
  picked?: boolean
  onOpen: () => void
  // one control at the end of the row, such as saving the message
  action?: ReactNode
  // marks the row for a list the arrow keys move through
  roving?: boolean
  // a key that acts on the row while the keyboard is on it
  shortcut?: string
  tabIndex?: number
}

// One message found in the archive: who it is from and when, then the message
// itself as the bubble it was. What opening it does is up to the list it is in.
export function MessageHit({
  message,
  index,
  term = '',
  picked,
  onOpen,
  action,
  roving,
  shortcut,
  tabIndex,
}: MessageHitProps) {
  const words = message.text || message.attachment || 'No text in this message'
  return (
    <div className="hit" data-picked={picked || undefined}>
      <button
        type="button"
        className="hit-open"
        data-roving={roving || undefined}
        aria-current={picked || undefined}
        aria-keyshortcuts={shortcut}
        tabIndex={tabIndex}
        onClick={onOpen}
      >
        {index === undefined ? null : <span className="hit-index">{index}</span>}
        <span className="hit-main">
          <span className="hit-meta">
            {messageSource(message)}, {formatDateTime(message.timestamp)}
          </span>
          <span className={`bubble clamp${isOutbound(message) ? ' you' : ''}`}>
            {marked(message, words, term).map((segment, at) =>
              segment.match ? <mark key={at}>{segment.text}</mark> : segment.text,
            )}
          </span>
        </span>
      </button>
      {action}
    </div>
  )
}
