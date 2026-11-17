import { useId, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { contactTitle } from '../../lib/conversations'
import { readShape, storeShape } from '../../lib/keptShape'
import { momentsOf, type Moment } from '../../lib/moments'
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

// the name the list's shape is remembered under, from one drawing to the next
const SHAPE = 'moments'
// with no shape remembered, what stands in for the list has this many cards
const CARDS_AT_FIRST = 3

// What is worth opening today. Each moment is a thing of its own, so each
// is its own card: who, when, and what was said or what it is.
export function Moments({
  memories,
  onRetry,
  contacts,
  onSelectContact,
  onOpenDay,
  onReplayFirstWords,
  onMakeCard,
}: MomentsProps) {
  const moments = useMemo(() => (memories ? momentsOf(memories) : []), [memories])
  const loaded = useMemo(() => new Map(contacts.map((contact) => [contact.chat_id, contact])), [contacts])
  const list = useRef<HTMLUListElement>(null)
  const answered = memories !== null
  const count = moments.length

  // The room the list takes is remembered as it was last drawn, so that
  // what stands in for it the next time takes the same room. An answer
  // with no moments is remembered as empty. No answer at all changes nothing.
  useLayoutEffect(() => {
    if (!answered) return
    const element = list.current
    if (!element) {
      storeShape(SHAPE, { rows: 0, height: 0 })
      return
    }
    // measured now, and again whenever the list changes size; a height that
    // has not changed is not handed on a second time
    let last = -1
    const keep = () => {
      const height = element.getBoundingClientRect().height
      if (height === last) return
      last = height
      storeShape(SHAPE, { rows: count, height })
    }
    keep()
    const observer = new ResizeObserver(keep)
    observer.observe(element)
    return () => observer.disconnect()
  }, [answered, count])

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

  // a moment is a place in time: it opens on its day. One from a build that
  // did not say the day opens its conversation.
  function open(moment: Moment) {
    const contact = contactFor(moment)
    if (!moment.day) return onSelectContact(contact)
    onOpenDay({
      contact,
      title: contactTitle(contact),
      day: moment.day,
      at: moment.at ?? null,
      quotedId: moment.quotedId,
    })
  }

  return (
    <section className="section ruled" aria-labelledby="moments-title">
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
        <ul ref={list} className="moment-cards" role="list">
          {moments.map((moment) => (
            <MomentCard key={moment.key} moment={moment} title={titleFor(moment)} onOpen={() => open(moment)} />
          ))}
        </ul>
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

function MomentCard({ moment, title, onOpen }: { moment: Moment; title: string; onOpen: () => void }) {
  const titleId = useId()
  const whenId = useId()
  const detailId = useId()
  return (
    <li>
      {/* the title is who it is with; when and what are read as its description */}
      <button
        type="button"
        className="row moment card"
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

// Stands in for the moments while they load. It takes the room the list
// took the last time it was drawn: as many cards, spread over the same
// height. So nothing on the page moves when the moments arrive: not the
// sections under them, and not the lists beside them, which are as long as
// these are tall. A day that had no moments is stood in for by the one line
// it had.
export function MomentsSkeleton() {
  // read once, when it is put on the page
  const [shape] = useState(() => readShape(SHAPE))
  const cards = shape ? shape.rows : CARDS_AT_FIRST

  return (
    <section className="section ruled" aria-hidden="true">
      <div className="section-head">
        <h2>Moments</h2>
        <span className="section-links">
          <span className="skeleton">Replay first words</span>
          <span className="skeleton">Make a card</span>
        </span>
      </div>
      {cards ? (
        <ul className="moment-cards" role="list" style={shape ? { minHeight: shape.height } : undefined}>
          {Array.from({ length: cards }, (_, card) => (
            <li key={card}>
              <div className="row moment card">
                <span className="moment-when">
                  <span className="skeleton">2 years ago today</span>
                </span>
                <span className="moment-main">
                  <span className="moment-name">
                    <span className="skeleton">Conversation name</span>
                  </span>
                  <span className="moment-detail">
                    <span className="skeleton">000 messages that day</span>
                  </span>
                </span>
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <p className="section-note">
          <span className="skeleton">Nothing from this day in earlier years</span>
        </p>
      )}
    </section>
  )
}
