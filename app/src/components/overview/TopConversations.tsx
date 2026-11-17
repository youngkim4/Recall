import { contactTitle, messageCount, receivedCount, sentCount } from '../../lib/conversations'
import { formatNumber, formatPercent } from '../../lib/format'
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

// The busiest conversations as a ranked table. Each row carries its count and
// its share of the whole archive; the bar is drawn against the largest and
// split into what you sent and what you received.
export function TopConversations({ ranked, archive, onSelectContact, onViewAll }: TopConversationsProps) {
  const rows = ranked.slice(0, SHOWN)
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
      <div className="card">
        <ol className="rows" role="list">
          {rows.map((contact, index) => {
            const count = messageCount(contact)
            const sent = sentCount(contact)
            const name = contactTitle(contact)
            const share = formatPercent(count / total)
            return (
              <li key={contact.chat_id}>
                <button
                  type="button"
                  className="row ranked-row"
                  aria-label={`${name}: ${formatNumber(count)} messages, ${share} of the archive, ${formatPercent(sent / Math.max(1, count))} sent by you`}
                  onClick={() => onSelectContact(contact)}
                >
                  <span className="ranked-place">{index + 1}</span>
                  <Truncate className="row-title">{name}</Truncate>
                  <span className="ranked-bar">
                    <VoiceBar sent={sent} received={receivedCount(contact)} scale={count / largest} />
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
