import { answerNotes } from '../../lib/answerNotes'
import { bookmarkId } from '../../lib/bookmarks'
import type { ChatMessage, Work } from '../../lib/chatStore'
import { plural } from '../../lib/format'
import type { SearchResult } from '../../types'
import { ChevronRightIcon } from '../Icons'
import { Disclosure } from '../ui/Disclosure'
import { MessageHit } from '../ui/MessageHit'
import { AnswerBody } from './AnswerBody'
import { WorkDone, Working } from './Working'

type ChatThreadProps = {
  messages: ChatMessage[]
  // messages newer than this arrive with a short rise; older ones are just there
  openedAt: number
  waiting: boolean
  // the steps of the question being answered, while the answer has not started
  work: Work | undefined
  // set where the cited messages stand in a pane beside the thread: the pane's
  // id, the answer whose messages it shows, and whether the pane is open (put
  // away, the control under an answer brings it back). Null where each
  // answer keeps its own under it.
  beside: { paneId: string; answerId: string; open: boolean } | null
  onShowCited: (answerId: string) => void
  // the cited message whose day is open, as bookmarkId gives it; '' for none
  openKey: string
  // `row` is the cite that was clicked, for its day to grow out of
  onOpenCitation: (citation: SearchResult, row: Element) => void
}

// message ids carry the time they were made: "user-<ms>-<random>"
function createdAt(message: ChatMessage) {
  return Number(message.id.split('-')[1]) || 0
}

export function ChatThread({
  messages,
  openedAt,
  waiting,
  work,
  beside,
  onShowCited,
  openKey,
  onOpenCitation,
}: ChatThreadProps) {
  // a thread that was opened arrives as one piece; in one that is being
  // started, each message arrives as it is sent
  const opened = messages.length > 0 && createdAt(messages[0]) <= openedAt
  return (
    <div className={opened ? 'chat-column arrive' : 'chat-column'}>
      {messages.map((message) => {
        const fresh = createdAt(message) > openedAt
        if (message.role === 'user') {
          return (
            <div key={message.id} className={`msg you${fresh ? ' is-new' : ''}`}>
              <p className="bubble you lg tail">{message.content}</p>
            </div>
          )
        }

        const citations = message.response?.citations ?? []
        const keywordOnly = message.response?.mode === 'local' && citations.length > 0
        return (
          <article
            key={message.id}
            className={`answer${fresh ? ' is-new' : ''}`}
            aria-busy={message.streaming || undefined}
          >
            {message.work ? <WorkDone work={message.work} /> : null}
            <AnswerBody text={message.content} streaming={message.streaming} />
            {keywordOnly ? (
              <p className="answer-note">AI answer unavailable. These are keyword matches.</p>
            ) : null}
            {answerNotes(message).map((note) => (
              <p key={note} className="answer-note">
                {note}
              </p>
            ))}
            {!citations.length ? null : beside ? (
              // the same words in the same place as the disclosure; here they put this answer's messages in the pane
              <button
                type="button"
                className="cites-show"
                aria-pressed={beside.open && message.id === beside.answerId}
                aria-controls={beside.paneId}
                onClick={() => onShowCited(message.id)}
              >
                {plural(citations.length, 'message')} cited
                <ChevronRightIcon />
              </button>
            ) : (
              <Disclosure className="cites" summary={`${plural(citations.length, 'message')} cited`}>
                <ol className="cite-list" role="list">
                  {citations.map((citation, index) => (
                    <li key={`${citation.chatId}-${citation.timestamp}-${index}`}>
                      <MessageHit
                        message={citation}
                        index={index + 1}
                        picked={bookmarkId(citation) === openKey}
                        onOpen={(row) => onOpenCitation(citation, row)}
                      />
                    </li>
                  ))}
                </ol>
              </Disclosure>
            )}
          </article>
        )
      })}

      {waiting ? <Working work={work} /> : null}
    </div>
  )
}
