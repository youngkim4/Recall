import { useRef } from 'react'
import { contactTitle, messageCount, voicesOf } from '../../lib/conversations'
import { formatNumber, formatPercent } from '../../lib/format'
import { useRowsThatFit } from '../../lib/useRowsThatFit'
import type { ArchiveSummary, Contact } from '../../types'
import { Truncate } from '../ui/Truncate'
import { VoiceBar, VoiceLegend } from '../ui/VoiceBar'

type TopConversationsProps = {
  // largest first
  ranked: Contact[]
  archive: ArchiveSummary
  onSelectContact: (contact: Contact) => void
  onViewAll: () => void
}

const SHOWN = 8
// beside a taller neighbor the list is as long as the room it is given, and never shorter than this
const FEWEST = 5
// a row is 12px above and below a 20px line, with a hairline over it
const rowHeight = (rem: number) => 2.75 * rem + 1

// The busiest conversations as a ranked table. Each row carries its count and
// its share of the whole archive; the bar is drawn against the largest and
// split into what you sent and what you received, where the list says which
// is which. Alone in a column it shows
// eight; in a card stretched to the height of what stands beside it, it shows
// as many as that height holds, so the card ends where its neighbor ends.
export function TopConversations({ ranked, archive, onSelectContact, onViewAll }: TopConversationsProps) {
  const card = useRef<HTMLDivElement>(null)
  const shown = useRowsThatFit(card, rowHeight, SHOWN, FEWEST)
  const rows = ranked.slice(0, shown)
  if (!rows.length) return null

  const largest = Math.max(1, messageCount(rows[0]))
  const total = Math.max(1, archive.messages)

  return (
    <section className="section" aria-labelledby="top-title">
      <div className="section-head">
        <h2 id="top-title">Most messages</h2>
        <VoiceLegend />
        <span className="section-links">
          <button type="button" className="link" onClick={onViewAll}>
            All {formatNumber(archive.conversations)} conversations
          </button>
        </span>
      </div>
      <div ref={card} className="card ranked-card">
        <ol className="rows" role="list">
          {rows.map((contact, index) => {
            const count = messageCount(contact)
            const voices = voicesOf(contact)
            const name = contactTitle(contact)
            const share = formatPercent(count / total)
            const yours = voices ? `, ${formatPercent(voices.sent / Math.max(1, count))} sent by you` : ''
            return (
              <li key={contact.chat_id}>
                <button
                  type="button"
                  className="row ranked-row"
                  aria-label={`${name}: ${formatNumber(count)} messages, ${share} of the archive${yours}`}
                  onClick={() => onSelectContact(contact)}
                >
                  <span className="ranked-place">{index + 1}</span>
                  <Truncate className="row-title">{name}</Truncate>
                  <span className="ranked-bar">
                    {voices ? (
                      <VoiceBar sent={voices.sent} received={voices.received} scale={count / largest} />
                    ) : (
                      <VoiceBar sent={0} received={0} unsaid={count} scale={count / largest} />
                    )}
                  </span>
                  <span className="ranked-count">{formatNumber(count)}</span>
                  <span className="ranked-share">{share}</span>
                </button>
              </li>
            )
          })}
        </ol>
      </div>
    </section>
  )
}
