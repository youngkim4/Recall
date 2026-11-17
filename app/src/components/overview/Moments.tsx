import { useId, useMemo } from 'react'
import { contactTitle } from '../../lib/conversations'
import { formatDate, plural } from '../../lib/format'
import type { Contact, MemoriesPayload } from '../../types'
import { Truncate } from '../ui/Truncate'

type MomentsProps = {
  // null when today's moments could not be loaded
  memories: MemoriesPayload | null
  onRetry: () => void
  contacts: Contact[]
  onSelectContact: (contact: Contact) => void
  // opens the day a moment is of, at the message it quotes
  onOpenDay: (day: MomentOfDay) => void
  onReplayFirstWords: () => void
  onMakeCard: () => void
}

// a moment as a place in time: one day of one conversation
export type MomentOfDay = {
  contact: Contact
  title: string
  // "YYYY-MM-DD"
  day: string
  // when the quoted message was sent, if the moment quotes one
  at: string | null
  // that message's id, where the server gives one
  quotedId?: string
}

type Moment = {
  key: string
  chatId: string
  // the saved name as the server sent it, or ''
  savedName: string
  // how this moment relates to today: "2 years ago today", "In 6 days"
  when: string
  // fromMe is null when the export does not say who sent the message
  quote?: { text: string; fromMe: boolean | null }
  // in a group, who wrote the quoted message
  quotedFrom?: string
  line: string
  // the day the moment is of, when its quoted message was sent, and that message's id
  day?: string
  at?: string | null
  quotedId?: string
}

function yearsAgoToday(years: number) {
  return years === 1 ? 'A year ago today' : `${years} years ago today`
}

function daysAway(days: number) {
  if (days === 0) return 'Today'
  return days === 1 ? 'Tomorrow' : `In ${days} days`
}

function buildMoments(memories: MemoriesPayload): Moment[] {
  const onThisDay = memories.onThisDay.slice(0, 3).map((item) => ({
    key: `day-${item.chatId}-${item.year}`,
    chatId: item.chatId,
    savedName: item.name,
    when: yearsAgoToday(item.yearsAgo),
    quote: item.preview ? { text: item.preview, fromMe: item.previewFromMe } : undefined,
    quotedFrom: item.previewSender || undefined,
    line: `${plural(item.count, 'message')} that day`,
    day: item.day,
    at: item.previewAt,
    quotedId: item.previewId || undefined,
  }))
  const anniversaries = memories.anniversaries.slice(0, 2).map((item) => ({
    key: `anniversary-${item.chatId}`,
    chatId: item.chatId,
    savedName: item.name,
    when: daysAway(item.inDays),
    line: `${plural(item.years, 'year')} since the first message`,
    // the day it began
    day: item.date.slice(0, 10),
  }))
  // "quiet" is measured to the end of the archive, so the date is given rather
  // than a length of time counted to today
  const quiet = memories.reconnect.slice(0, 2).map((item) => ({
    key: `quiet-${item.chatId}`,
    chatId: item.chatId,
    savedName: item.name,
    when: `Quiet since ${formatDate(item.lastDate)}`,
    line: plural(item.count, 'message'),
    // the last day anything was said
    day: item.lastDate.slice(0, 10),
  }))
  return [...onThisDay, ...anniversaries, ...quiet]
}

export function Moments({
  memories,
  onRetry,
  contacts,
  onSelectContact,
  onOpenDay,
  onReplayFirstWords,
  onMakeCard,
}: MomentsProps) {
  const moments = useMemo(() => (memories ? buildMoments(memories) : []), [memories])
  const loaded = useMemo(() => new Map(contacts.map((contact) => [contact.chat_id, contact])), [contacts])

  // a moment can come from a conversation outside the loaded list; it is named
  // and opened the same way
  function contactFor(moment: Moment): Contact {
    return loaded.get(moment.chatId) ?? { chat_id: moment.chatId, display_name: moment.savedName }
  }

  // "Sam in Family", the way a cited message names its source
  function titleFor(moment: Moment) {
    const conversation = contactTitle(contactFor(moment))
    return moment.quotedFrom ? `${moment.quotedFrom} in ${conversation}` : conversation
  }

  return (
    <section className="section" aria-labelledby="moments-title">
      <div className="section-head">
        <h2 id="moments-title">Moments</h2>
        <span className="section-links">
          <button type="button" className="link" onClick={onReplayFirstWords}>
            Replay first words
          </button>
          <button type="button" className="link" onClick={onMakeCard}>
            Make a card
          </button>
        </span>
      </div>
      {moments.length ? (
        <div className="card">
          <ul className="rows moments" role="list">
            {moments.map((moment) => (
              <MomentRow
                key={moment.key}
                moment={moment}
                title={titleFor(moment)}
                // a moment is a place in time: it opens on its day. One from a
                // build that did not say the day opens its conversation.
                onOpen={() =>
                  moment.day
                    ? onOpenDay({
                        contact: contactFor(moment),
                        title: contactTitle(contactFor(moment)),
                        day: moment.day,
                        at: moment.at ?? null,
                        quotedId: moment.quotedId,
                      })
                    : onSelectContact(contactFor(moment))
                }
              />
            ))}
          </ul>
        </div>
      ) : memories ? (
        <p className="section-note">Nothing from this day in earlier years</p>
      ) : (
        <p className="section-note">
          Today’s moments could not be loaded.{' '}
          <button type="button" className="link in-text" onClick={onRetry}>
            Try again
          </button>
        </p>
      )}
    </section>
  )
}

function MomentRow({ moment, title, onOpen }: { moment: Moment; title: string; onOpen: () => void }) {
  const titleId = useId()
  const whenId = useId()
  const detailId = useId()
  return (
    <li>
      {/* the title is who it is with; when and what are read as its description */}
      <button
        type="button"
        className="row moment"
        aria-labelledby={titleId}
        aria-describedby={`${whenId} ${detailId}`}
        onClick={onOpen}
      >
        <span className="moment-when" id={whenId}>
          {moment.when}
        </span>
        <span className="moment-main">
          <Truncate className="moment-name" id={titleId}>
            {title}
          </Truncate>
          <span className="moment-detail" id={detailId}>
            {moment.quote ? <MomentQuote text={moment.quote.text} fromMe={moment.quote.fromMe} /> : null}
            <span className="moment-line">{moment.line}</span>
          </span>
        </span>
      </button>
    </li>
  )
}

// Blue is what you sent and gray is what you received, here as everywhere.
// When the export does not say, the words are quoted without taking a side.
function MomentQuote({ text, fromMe }: { text: string; fromMe: boolean | null }) {
  if (fromMe === null) return <span className="moment-line">“{text}”</span>
  return (
    <span className={`bubble clamp${fromMe ? ' you' : ''}`}>
      {fromMe ? <span className="sr-only">You wrote: </span> : null}
      {text}
    </span>
  )
}

// stands in for the list while it loads; the sections below wait with it, so
// nothing already on the page is pushed down when the moments arrive
export function MomentsSkeleton() {
  return (
    <section className="section" aria-hidden="true">
      <div className="section-head">
        <h2>Moments</h2>
      </div>
      <div className="card">
        <ul className="rows moments" role="list">
          {[0, 1, 2].map((row) => (
            <li key={row}>
              <div className="row moment">
                <span>
                  <span className="skeleton">2 years ago today</span>
                </span>
                <span className="moment-main">
                  <span>
                    <span className="skeleton">Conversation name</span>
                  </span>
                  <span>
                    <span className="skeleton">000 messages that day</span>
                  </span>
                </span>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}
