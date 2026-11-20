import { bookmarkId } from '../../lib/bookmarks'
import { plural } from '../../lib/format'
import { rovingKeyDown } from '../../lib/roving'
import type { SearchResult } from '../../types'
import { CloseIcon } from '../Icons'
import { MessageHit } from '../ui/MessageHit'

type ChatCitesProps = {
  // names the pane for the controls in the thread that fill it
  id: string
  // the answer these belong to; another answer brings its messages in afresh
  answerId: string
  citations: SearchResult[]
  // the cited message whose day is open, as bookmarkId gives it; '' for none
  openKey: string
  // `row` is the cite that was clicked, for its day to grow out of
  onOpen: (citation: SearchResult, row: Element) => void
  // put away: still on the page while its pane closes, but out of reach
  closed: boolean
  onClose: () => void
}

// The messages one answer cites, beside the thread in a wide window: how many,
// then each as the message it was, under the number the answer calls it by.
// The list is one stop for Tab; the arrow keys move through it, and Enter
// opens the day around a message.
export function ChatCites({ id, answerId, citations, openKey, onOpen, closed, onClose }: ChatCitesProps) {
  return (
    <aside id={id} className="pane chat-cites" aria-label="Cited messages" inert={closed}>
      <div className="pane-head chat-cites-head">
        <h2 key={answerId} className="swap">
          {plural(citations.length, 'message')} cited
        </h2>
        <button type="button" className="icon-btn" aria-label="Close cited messages" title="Close" onClick={onClose}>
          <CloseIcon />
        </button>
      </div>
      {/* a new list starts from its top, whatever the last one was scrolled to */}
      <div key={answerId} className="pane-scroll">
        <ol className="chat-cites-list arrive" role="list" onKeyDown={rovingKeyDown}>
          {citations.map((citation, index) => (
            <li key={`${bookmarkId(citation)}-${index}`}>
              <MessageHit
                message={citation}
                index={index + 1}
                picked={bookmarkId(citation) === openKey}
                onOpen={(row) => onOpen(citation, row)}
                roving
                tabIndex={index === 0 ? 0 : -1}
              />
            </li>
          ))}
        </ol>
      </div>
    </aside>
  )
}
