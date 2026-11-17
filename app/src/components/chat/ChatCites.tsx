import { bookmarkId } from '../../lib/bookmarks'
import { plural } from '../../lib/format'
import { rovingKeyDown } from '../../lib/roving'
import type { SearchResult } from '../../types'
import { MessageHit } from '../ui/MessageHit'

type ChatCitesProps = {
  // names the pane for the controls in the thread that fill it
  id: string
  // the answer these belong to; another answer brings its messages in afresh
  answerId: string
  citations: SearchResult[]
  // the cited message whose day is open, as bookmarkId gives it; '' for none
  openKey: string
  onOpen: (citation: SearchResult) => void
}

// The messages one answer cites, beside the thread in a wide window: how many,
// then each as the message it was, under the number the answer calls it by.
// The list is one stop for Tab; the arrow keys move through it, and Enter
// opens the day around a message.
export function ChatCites({ id, answerId, citations, openKey, onOpen }: ChatCitesProps) {
  return (
    <aside id={id} className="pane chat-cites" aria-label="Cited messages">
      <div className="pane-head">
        <h2 key={answerId} className="swap">
          {plural(citations.length, 'message')} cited
        </h2>
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
                onOpen={() => onOpen(citation)}
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
