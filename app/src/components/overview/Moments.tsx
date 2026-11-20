import { useId, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { readShape, storeShape } from '../../lib/keptShape'
import { momentTitle, openMoment, type MomentOfDay } from '../../lib/momentOpen'
import { shelfOf, type Moment } from '../../lib/moments'
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

export type { MomentOfDay } from '../../lib/momentOpen'

// the name the shelf's shape is remembered under, from one drawing to the next
const SHAPE = 'moments'
// with no shape remembered, what stands in for the shelf has this many cards: three rows
const CARDS_AT_FIRST = 9

// What is worth opening today, as one shelf of cards, after Spotify's home:
// what was said on this day in earlier years, then the anniversaries that
// are near, three cards a row. Each card is who, when, what was said, and
// how much was said that day; it opens the day.
export function Moments({
  memories,
  onRetry,
  contacts,
  onSelectContact,
  onOpenDay,
  onReplayFirstWords,
  onMakeCard,
}: MomentsProps) {
  const moments = useMemo(() => (memories ? shelfOf(memories) : []), [memories])
  const loaded = useMemo(() => new Map(contacts.map((contact) => [contact.chat_id, contact])), [contacts])
  const list = useRef<HTMLUListElement>(null)
  const answered = memories !== null
  const count = moments.length

  // The room the shelf takes is remembered as it was last drawn, so that
  // what stands in for it the next time takes the same room.
  useLayoutEffect(() => {
    if (!answered) return
    const element = list.current
    if (!element) {
      storeShape(SHAPE, { rows: 0, height: 0 })
      return
    }
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
        <ul ref={list} className="moment-shelf" role="list">
          {moments.map((moment) => (
            <MomentCard
              key={moment.key}
              moment={moment}
              title={momentTitle(moment, loaded)}
              onOpen={(card) => openMoment(moment, loaded, onOpenDay, onSelectContact, card)}
            />
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

function MomentCard({ moment, title, onOpen }: { moment: Moment; title: string; onOpen: (card: Element) => void }) {
  const titleId = useId()
  const whenId = useId()
  const detailId = useId()
  return (
    <li>
      {/* the title is who it is with; when and what are read as its description */}
      <button
        type="button"
        className="moment-card card"
        aria-labelledby={titleId}
        aria-describedby={`${whenId} ${detailId}`}
        onClick={(event) => onOpen(event.currentTarget)}
      >
        <span className="moment-head">
          <Truncate className="moment-name" id={titleId}>
            {title}
          </Truncate>
          <span className="moment-when" id={whenId}>
            {moment.when}
          </span>
        </span>
        <span className="moment-detail" id={detailId}>
          {moment.quote ? <MomentQuote text={moment.quote.text} fromMe={moment.quote.fromMe} /> : null}
          <span className="moment-line">{moment.line}</span>
        </span>
      </button>
    </li>
  )
}

// Blue is what you sent and gray is what you received, here as everywhere.
// When the export does not say, the words are quoted without taking a side.
function MomentQuote({ text, fromMe }: { text: string; fromMe: boolean | null }) {
  if (fromMe === null) return <span className="moment-quote">“{text}”</span>
  return (
    <span className={`bubble clamp${fromMe ? ' you' : ''}`}>
      {fromMe ? <span className="sr-only">You wrote: </span> : null}
      {text}
    </span>
  )
}

// Stands in for the shelf while it loads, as many cards as it last had, so
// nothing under it moves when the moments arrive.
export function MomentsSkeleton() {
  const [shape] = useState(() => readShape(SHAPE))
  const cards = shape ? shape.rows : CARDS_AT_FIRST
  return (
    <section className="section" aria-hidden="true">
      <div className="section-head">
        <h2>Moments</h2>
        <span className="section-links">
          <span className="skeleton">Replay first words</span>
          <span className="skeleton">Make a card</span>
        </span>
      </div>
      {cards ? (
        <ul className="moment-shelf" role="list" style={shape ? { minHeight: shape.height } : undefined}>
          {Array.from({ length: cards }, (_, card) => (
            <li key={card}>
              <div className="moment-card card">
                <span className="moment-head">
                  <span className="moment-name">
                    <span className="skeleton">Conversation name</span>
                  </span>
                </span>
                <span className="moment-detail">
                  <span className="skeleton">000 messages that day</span>
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
