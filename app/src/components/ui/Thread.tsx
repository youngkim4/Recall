import { parseDate } from '../../lib/format'
import { ImageIcon, PaperclipIcon } from '../Icons'

export type ThreadMessage = {
  key: string
  timestamp: string
  // empty when the message was only an attachment
  text: string
  // "Photo", "3 photos": what the message carried, if anything
  attachment?: string
  // null when the export does not say who sent it
  fromMe: boolean | null
  // who wrote a received message, where that is not already plain
  sender?: string
}

const dayFormat = new Intl.DateTimeFormat(undefined, {
  weekday: 'short',
  month: 'short',
  day: 'numeric',
  year: 'numeric',
})

function dayOf(timestamp: string) {
  const date = parseDate(timestamp)
  return date ? dayFormat.format(date) : ''
}

const timeFormat = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' })
// a pause this long between two messages gets the time written after it
const PAUSE_MS = 15 * 60 * 1000
const PICTURES = /photo|gif/i

// how long passed between two messages; none when either time cannot be read
function pause(earlier: string, later: string) {
  const from = parseDate(earlier)
  const to = parseDate(later)
  return from && to ? to.valueOf() - from.valueOf() : 0
}

type ThreadProps = {
  messages: ThreadMessage[]
  // Within one day the date is known, so the time is written instead: at the
  // first message and after every long pause.
  times?: boolean
  // the one message the thread was opened for
  pickedKey?: string
}

// Messages the way Messages shows them: yours in blue on the right, theirs in
// gray on the left, a date between days. Consecutive messages from one person
// sit close together and the last of them takes the tail.
export function Thread({ messages, times = false, pickedKey }: ThreadProps) {
  return (
    <div className="thread">
      {messages.map((message, index) => {
        const before = messages[index - 1]
        const after = messages[index + 1]
        const day = dayOf(message.timestamp)
        const at = parseDate(message.timestamp)
        const pauseBefore = !before || pause(before.timestamp, message.timestamp) >= PAUSE_MS
        const pauseAfter = Boolean(after) && pause(message.timestamp, after.timestamp) >= PAUSE_MS
        // the line above a message: the time after a pause, or the date on a new day
        const timeMark = at && pauseBefore ? timeFormat.format(at) : ''
        const dayMark = !before || dayOf(before.timestamp) !== day ? day : ''
        const mark = times ? timeMark : dayMark
        const newDay = Boolean(mark)
        const sameVoice = (other?: ThreadMessage) =>
          Boolean(other) && other?.fromMe === message.fromMe && other?.sender === message.sender
        const startsTurn = newDay || !sameVoice(before)
        const endsTurn = !after || (times && pauseAfter) || dayOf(after.timestamp) !== day || !sameVoice(after)
        const side = message.fromMe ? ' you' : ''

        return (
          <div key={message.key} className="thread-item" data-picked={message.key === pickedKey || undefined}>
            {mark ? <p className="stamp">{mark}</p> : null}
            <div className={`msg${side}${startsTurn && !newDay ? ' turn' : ''}`}>
              {startsTurn && !message.fromMe && message.sender ? (
                <span className="msg-sender">{message.sender}</span>
              ) : null}
              {message.attachment ? (
                <span className={`bubble attachment${side}${endsTurn && !message.text ? ' tail' : ''}`}>
                  {PICTURES.test(message.attachment) ? <ImageIcon /> : <PaperclipIcon />}
                  {message.attachment}
                </span>
              ) : null}
              {message.text || !message.attachment ? (
                <span className={`bubble${side}${endsTurn ? ' tail' : ''}`}>
                  {message.fromMe ? <span className="sr-only">You: </span> : null}
                  {message.text || 'No text'}
                </span>
              ) : null}
            </div>
          </div>
        )
      })}
    </div>
  )
}
