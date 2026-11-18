import { useMemo, useState } from 'react'
import type { ActivityPoint } from '../../lib/activity'
import { messageCount } from '../../lib/conversations'
import { formatDate, formatNumber, formatPercent, parseDate } from '../../lib/format'
import { topShares } from '../../lib/spark'
import type { ArchiveSummary, Contact } from '../../types'
import { LeadChart } from './LeadChart'

type ArchiveTilesProps = {
  archive: ArchiveSummary
  // every month of the archive in both voices; null while it is read, and when it could not be
  months: ActivityPoint[] | null
  // true while the months are still being read
  waiting: boolean
  // busiest first
  ranked: Contact[]
  onOpenSetup: () => void
}

// how many conversations the third figure sets against the rest
const TOP = 5
const DAY_MS = 86_400_000

// how long ago, in the largest unit that says it in one number
function ago(days: number) {
  if (days < 1) return 'Today'
  if (days < 60) return `${formatNumber(days)} ${days === 1 ? 'day' : 'days'} ago`
  const months = Math.round(days / 30.44)
  if (months < 24) return `${months} months ago`
  return `${Math.round(months / 12)} years ago`
}

// The archive in four figures. The first, how many messages there are,
// leads: it is set large over half the row, its name and figure in the
// middle of the row's height, with a chart beside it of how those messages
// came, a line through the months. The other three each have one line of words under them and no
// chart: a small chart under every figure is the row of identical stat
// tiles that reads as generated.
export function ArchiveTiles({ archive, months, waiting, ranked, onOpenSetup }: ArchiveTilesProps) {
  // the day the page was opened on
  const [now] = useState(() => Date.now())
  const sent = useMemo(() => (months ?? []).reduce((sum, month) => sum + month.sent, 0), [months])
  const received = useMemo(() => (months ?? []).reduce((sum, month) => sum + month.received, 0), [months])
  const shares = useMemo(
    () => topShares(ranked.slice(0, TOP).map(messageCount), archive.messages, TOP),
    [ranked, archive.messages],
  )
  const last = parseDate(archive.lastMessage)
  // how many days lie between the newest message and today
  const behind = last ? Math.max(0, Math.floor((now - last.valueOf()) / DAY_MS)) : 0
  const held = shares.reduce((sum, share) => sum + share, 0)

  return (
    <section className="tiles" aria-label="The archive in figures">
      <div className="tile tile-lead">
        <p className="tile-label">Messages</p>
        <p className="tile-value">{formatNumber(archive.messages)}</p>
        {months && months.length > 1 ? (
          <LeadChart months={months} />
        ) : waiting ? (
          <span className="skeleton lead-chart lead-waiting" />
        ) : null}
      </div>

      <div className="tile">
        <p className="tile-label">You wrote</p>
        <p className="tile-value">
          {sent + received > 0 ? (
            formatPercent(sent / (sent + received))
          ) : waiting ? (
            <span className="skeleton">00%</span>
          ) : (
            ''
          )}
        </p>
        <p className="tile-note">{sent + received > 0 ? `${formatNumber(sent)} messages` : ''}</p>
      </div>

      <div className="tile">
        <p className="tile-label">Conversations</p>
        <p className="tile-value">{formatNumber(archive.conversations)}</p>
        <p className="tile-note">
          {shares.length > 1 ? `Top ${shares.length} hold ${formatPercent(held)}` : ''}
        </p>
      </div>

      <div className="tile">
        <p className="tile-label">Imported through</p>
        <p className="tile-value">{last ? formatDate(last) : ''}</p>
        <p className="tile-note tile-action">
          <span>{last ? ago(behind) : ''}</span>
          <button type="button" className="link" onClick={onOpenSetup}>
            Update
          </button>
        </p>
      </div>
    </section>
  )
}

// holds the place of the tiles while the archive is first read
export function ArchiveTilesSkeleton() {
  return (
    <div className="tiles" aria-hidden="true">
      {['Messages', 'You wrote', 'Conversations', 'Imported through'].map((label) => (
        <div key={label} className={label === 'Messages' ? 'tile tile-lead' : 'tile'}>
          <p className="tile-label">{label}</p>
          <p className="tile-value">
            <span className="skeleton">000,000</span>
          </p>
          {label === 'Messages' ? null : (
            <p className="tile-note">
              <span className="skeleton">000 a day</span>
            </p>
          )}
        </div>
      ))}
    </div>
  )
}
