import type { ChatMessage } from '../../lib/chatStore'
import { plural } from '../../lib/format'
import { Disclosure } from '../ui/Disclosure'
import { MessageHit } from '../ui/MessageHit'
import { AnswerBody } from './AnswerBody'

type ChatThreadProps = {
  messages: ChatMessage[]
  // messages newer than this arrive with a short rise; older ones are just there
  openedAt: number
  waiting: boolean
  waitingStatus: string
  onOpenCitation: (chatId: string) => void
}

// message ids carry the time they were made: "user-<ms>-<random>"
function createdAt(message: ChatMessage) {
  return Number(message.id.split('-')[1]) || 0
}

export function ChatThread({ messages, openedAt, waiting, waitingStatus, onOpenCitation }: ChatThreadProps) {
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
          <article key={message.id} className="answer" aria-busy={message.streaming || undefined}>
            <AnswerBody text={message.content} streaming={message.streaming} />
            {keywordOnly ? (
              <p className="answer-note">AI answer unavailable. These are keyword matches.</p>
            ) : null}
            {citations.length ? (
              <Disclosure className="cites" summary={`${plural(citations.length, 'message')} cited`}>
                <ol className="cite-list" role="list">
                  {citations.map((citation, index) => (
                    <li key={`${citation.chatId}-${citation.timestamp}-${index}`}>
                      <MessageHit message={citation} index={index + 1} onOpen={() => onOpenCitation(citation.chatId)} />
                    </li>
                  ))}
                </ol>
              </Disclosure>
            ) : null}
          </article>
        )
      })}

      {waiting ? (
        <div className="chat-waiting is-new" role="status">
          <span className="typing" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          <span>{waitingStatus || 'Thinking…'}</span>
        </div>
      ) : null}
    </div>
  )
}
