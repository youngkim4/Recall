import { useId, useMemo, useState } from 'react'
import { conversationName } from '../../lib/conversations'
import { formatNumber, formatPercent, plural } from '../../lib/format'
import type { PreviewPayload } from '../../types'
import { Truncate } from './Truncate'
import { VoiceBar } from './VoiceBar'

// how many people the list shows before it is opened; your own row is shown as well
const SHOWN = 5
// a list this short is shown whole
const SHOWN_WHOLE = 8

type Person = {
  key: string
  name: string
  count: number
  you: boolean
  // null when the place cannot be known
  place: number | null
}

type Ranking = {
  people: Person[]
  // people who were counted but are too many to list, and what they wrote
  unlisted: { people: number; messages: number } | null
}

function rankPeople(preview: PreviewPayload): Ranking {
  const speakers = preview.dynamics?.speakers ?? []
  const sent = preview.stats.sentCount ?? 0
  const received = preview.stats.receivedCount ?? 0
  const listed = speakers.reduce((sum, speaker) => sum + speaker.count, 0)
  const unlistedPeople = Math.max(0, (preview.dynamics?.speakerCount ?? speakers.length) - speakers.length)
  const fewestListed = speakers.length ? speakers[speakers.length - 1].count : 0
  const people = [
    { key: 'you', name: 'You', count: sent, you: true },
    ...speakers.map((speaker) => ({
      key: speaker.handle,
      name: conversationName(speaker.name, speaker.handle),
      count: speaker.count,
      you: false,
    })),
  ]
    .sort((a, b) => b.count - a.count)
    .map((person, index) => ({
      ...person,
      // with people left off the list, a place below the last of them is not known
      place: unlistedPeople > 0 && person.count < fewestListed ? null : index + 1,
    }))
  return {
    people,
    unlisted: unlistedPeople > 0 ? { people: unlistedPeople, messages: Math.max(0, received - listed) } : null,
  }
}

// The list in four parts: the busiest, the rows between them and you, your
// own row, and the rows after it. The second and fourth open and close.
function partsOf(people: Person[]) {
  if (people.length <= SHOWN_WHOLE) return { top: people, between: [], own: [], after: [] }
  const you = people.findIndex((person) => person.you)
  if (you < SHOWN) return { top: people.slice(0, SHOWN), between: people.slice(SHOWN), own: [], after: [] }
  return {
    top: people.slice(0, SHOWN),
    between: people.slice(SHOWN, you),
    own: [people[you]],
    after: people.slice(you + 1),
  }
}

// A group has more than two voices: you, and each person who wrote in it.
// Everyone has a true place. The list starts with the busiest and your own
// row; opened, the rest fill in around it.
export function People({ preview }: { preview: PreviewPayload }) {
  const [open, setOpen] = useState(false)
  const titleId = useId()
  const betweenId = useId()
  const afterId = useId()
  const { people, unlisted } = useMemo(() => rankPeople(preview), [preview])
  const { top, between, own, after } = useMemo(() => partsOf(people), [people])
  if (people.length < 2) return null

  const largest = Math.max(1, ...people.map((person) => person.count))
  const total = Math.max(1, (preview.stats.sentCount ?? 0) + (preview.stats.receivedCount ?? 0))
  const folded = between.length + after.length > 0

  const rows = (list: Person[]) => (
    <ol className="rows" role="list">
      {list.map((person) => (
        <li key={person.key} className="row ranked-row">
          <span className="ranked-place">{person.place}</span>
          <Truncate className="row-title">{person.name}</Truncate>
          <span className="ranked-bar">
            <VoiceBar
              sent={person.you ? person.count : 0}
              received={person.you ? 0 : person.count}
              scale={person.count / largest}
            />
          </span>
          <span className="ranked-count">
            {formatNumber(person.count)}
            <span className="sr-only"> messages</span>
          </span>
          <span className="ranked-share">{formatPercent(person.count / total)}</span>
        </li>
      ))}
    </ol>
  )

  const fold = (id: string, list: Person[]) =>
    list.length ? (
      <div id={id} className={`disclosure-body${open ? ' is-open' : ''}`} inert={!open}>
        <div>{rows(list)}</div>
      </div>
    ) : null

  return (
    <section className="section people" aria-labelledby={titleId}>
      <div className="section-head">
        <h3 id={titleId}>People</h3>
        {folded ? (
          <span className="section-links">
            <button
              type="button"
              className="link"
              aria-expanded={open}
              aria-controls={[between.length ? betweenId : '', after.length ? afterId : ''].join(' ').trim()}
              onClick={() => setOpen((current) => !current)}
            >
              {open ? 'Show fewer' : `Show all ${formatNumber(people.length)}`}
            </button>
          </span>
        ) : null}
      </div>
      {rows(top)}
      {fold(betweenId, between)}
      {own.length ? (
        // closed, a line says that places are skipped between the row above and yours
        <div className={`people-own${between.length && !open ? ' is-apart' : ''}`}>{rows(own)}</div>
      ) : null}
      {fold(afterId, after)}
      {unlisted ? (
        <p className="section-note">
          {plural(unlisted.people, 'more person', 'more people')} wrote {plural(unlisted.messages, 'message')}
        </p>
      ) : null}
    </section>
  )
}
