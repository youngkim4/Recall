import { useMemo, useState, type CSSProperties } from 'react'
import { momentTitle, openMoment, type MomentOfDay } from '../../lib/momentOpen'
import { quietOf } from '../../lib/moments'
import { LIST_FEWEST, listRowHeight } from '../../lib/overviewFit'
import { useRowsThatFit } from '../../lib/useRowsThatFit'
import type { Contact, MemoriesPayload } from '../../types'
import { Truncate } from '../ui/Truncate'

type GoneQuietProps = {
  // null while they load, or when they could not be
  memories: MemoriesPayload | null
  // true when they could not be: the moments above say so, and offer to try again
  failed: boolean
  contacts: Contact[]
  onOpenDay: (day: MomentOfDay) => void
  onSelectContact: (contact: Contact) => void
}

// The conversations that have gone quiet, beside the people of lately: who,
// since when, and how much was said before. Each opens on its last day. It
// shows as many as reach the foot of the window (overviewFit.ts); "All" opens
// the rest in place, where there are more than fit.
export function GoneQuiet({ memories, failed, contacts, onOpenDay, onSelectContact }: GoneQuietProps) {
  const [all, setAll] = useState(false)
  const [box, fit] = useRowsThatFit(listRowHeight, LIST_FEWEST, LIST_FEWEST)
  const total = memories?.reconnect.length ?? 0
  const more = total > fit
  const count = all ? total : fit
  const quiet = useMemo(() => (memories ? quietOf(memories, count) : null), [memories, count])
  const loaded = useMemo(() => new Map(contacts.map((contact) => [contact.chat_id, contact])), [contacts])
  const shown = quiet ?? Array.from({ length: fit }, () => null)
  return (
    <section className="section" aria-labelledby="quiet-title">
      <div className="section-head">
        <h2 id="quiet-title">Gone quiet</h2>
        {more || all ? (
          <span className="section-links">
            <button type="button" className="link" aria-expanded={all} onClick={() => setAll((open) => !open)}>
              {all ? 'Fewer' : 'All'}
            </button>
          </span>
        ) : null}
      </div>
      {failed ? (
        <p className="section-note">Could not be loaded</p>
      ) : (
        // opened whole, the list grows past the window and the page scrolls
        <div ref={box} className="overview-fit" data-all={all || undefined}>
          <ol
            className="rows ruled overview-list"
            role="list"
            aria-busy={!quiet || undefined}
            style={{ '--rows': Math.max(1, shown.length) } as CSSProperties}
          >
            {shown.map((moment, index) =>
              moment ? (
                <li key={moment.key}>
                  <button
                    type="button"
                    className="row quiet-row"
                    onClick={(event) => openMoment(moment, loaded, onOpenDay, onSelectContact, event.currentTarget)}
                  >
                    <Truncate className="row-title">{momentTitle(moment, loaded)}</Truncate>
                    <span className="quiet-since">{moment.when}</span>
                    <span className="quiet-count">{moment.line}</span>
                  </button>
                </li>
              ) : (
                <li key={index}>
                  <div className="row quiet-row" aria-hidden="true">
                    <span>
                      <span className="skeleton">Conversation name</span>
                    </span>
                  </div>
                </li>
              ),
            )}
            {quiet && !quiet.length ? <li className="pane-note">No conversation has gone quiet</li> : null}
          </ol>
        </div>
      )}
    </section>
  )
}
