import type { ReactNode } from 'react'
import { isOutbound, messageSource } from '../../lib/conversations'
import { formatDateTime } from '../../lib/format'
import { splitByTerm } from '../../lib/highlight'
import type { SearchResult } from '../../types'

type MessageHitProps = {
  message: SearchResult
  // its place in a numbered list, when it is in one
  index?: number
  // words to pick out in the text
  term?: string
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
// itself as the bubble it was. Opening it goes to its conversation.
export function MessageHit({ message, index, term = '', onOpen, action, roving, shortcut, tabIndex }: MessageHitProps) {
  const words = message.text || message.attachment || 'No text in this message'
  return (
    <div className="hit">
      <button
        type="button"
        className="hit-open"
        data-roving={roving || undefined}
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
            {splitByTerm(words, message.text ? term : '').map((segment, at) =>
              segment.match ? <mark key={at}>{segment.text}</mark> : segment.text,
            )}
          </span>
        </span>
      </button>
      {action}
    </div>
  )
}
