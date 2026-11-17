import { useMemo, useState } from 'react'
import type { ActivityPoint } from '../../lib/activity'
import { messageCount } from '../../lib/conversations'
import { daysSpanned } from '../../lib/days'
import { formatDate, formatNumber, formatPercent, parseDate } from '../../lib/format'
import { sparkArea, sparkLine, topShares } from '../../lib/spark'
import type { ArchiveSummary, Contact } from '../../types'
import { VoiceBar } from '../ui/VoiceBar'

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

// the box the line of months is drawn in; it is stretched to the tile
const SPARK = { width: 120, height: 24 }
// how many conversations the third tile sets against the rest
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

// The archive in four figures, each over a small picture of itself: how much
// it holds and how that built up, how much of it is yours, how much of it sits
// in a few conversations, and how far it reaches.
export function ArchiveTiles({ archive, months, waiting, ranked, onOpenSetup }: ArchiveTilesProps) {
  // the day the page was opened on
  const [now] = useState(() => Date.now())
  const totals = useMemo(() => (months ?? []).map((month) => month.total), [months])
  const sent = useMemo(() => (months ?? []).reduce((sum, month) => sum + month.sent, 0), [months])
  const received = useMemo(() => (months ?? []).reduce((sum, month) => sum + month.received, 0), [months])
  const shares = useMemo(
    () => topShares(ranked.slice(0, TOP).map(messageCount), archive.messages, TOP),
    [ranked, archive.messages],
  )
  const days = daysSpanned(archive.firstMessage, archive.lastMessage)
  const first = parseDate(archive.firstMessage)
  const last = parseDate(archive.lastMessage)
  // how much of the time from the first message to today the archive holds,
  // and how many days lie between its newest message and today
  const reach = first && last ? (last.valueOf() - first.valueOf()) / Math.max(DAY_MS, now - first.valueOf()) : 0
  const behind = last ? Math.max(0, Math.floor((now - last.valueOf()) / DAY_MS)) : 0
  const held = shares.reduce((sum, share) => sum + share, 0)

  return (
    <section className="tiles" aria-label="The archive in figures">
      <div className="tile card">
        <p className="tile-label">Messages</p>
        <p className="tile-value">{formatNumber(archive.messages)}</p>
        <div className="tile-picture" aria-hidden="true">
          {totals.length ? (
            <svg className="spark" viewBox={`0 0 ${SPARK.width} ${SPARK.height}`} preserveAspectRatio="none">
              <path className="spark-area" d={sparkArea(totals, SPARK.width, SPARK.height)} />
              <path className="spark-line" d={sparkLine(totals, SPARK.width, SPARK.height)} />
            </svg>
          ) : waiting ? (
            <span className="skeleton tile-waiting" />
          ) : null}
        </div>
        <p className="tile-note">{days ? `${formatNumber(Math.round(archive.messages / days))} a day` : ''}</p>
      </div>

      <div className="tile card">
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
        <div className="tile-picture" aria-hidden="true">
          {sent + received > 0 ? (
            <VoiceBar sent={sent} received={received} />
          ) : waiting ? (
            <span className="skeleton tile-waiting" />
          ) : null}
        </div>
        <p className="tile-note">{sent + received > 0 ? `${formatNumber(sent)} messages` : ''}</p>
      </div>

      <div className="tile card">
        <p className="tile-label">Conversations</p>
        <p className="tile-value">{formatNumber(archive.conversations)}</p>
        <div className="tile-picture" aria-hidden="true">
          <span className="tile-parts">
            {shares.map((share, index) => (
              <i key={index} data-part={index} style={{ flexGrow: share }} />
            ))}
            {held < 1 ? <i data-part="rest" style={{ flexGrow: 1 - held }} /> : null}
          </span>
        </div>
        <p className="tile-note">
          {shares.length > 1 ? `Top ${shares.length} hold ${formatPercent(held)}` : ''}
        </p>
      </div>

      <div className="tile card">
        <p className="tile-label">Imported through</p>
        <p className="tile-value">{last ? formatDate(last) : ''}</p>
        <div className="tile-picture" aria-hidden="true">
          <span className="tile-parts">
            <i data-part="0" style={{ flexGrow: Math.max(0, Math.min(1, reach)) }} />
            {reach < 1 ? <i data-part="rest" style={{ flexGrow: 1 - Math.max(0, Math.min(1, reach)) }} /> : null}
          </span>
        </div>
        {/* the empty end of the bar is the time since; the way to close it is beside it */}
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
        <div key={label} className="tile card">
          <p className="tile-label">{label}</p>
          <p className="tile-value">
            <span className="skeleton">000,000</span>
          </p>
          <div className="tile-picture">
            <span className="skeleton tile-waiting" />
          </div>
          <p className="tile-note">
            <span className="skeleton">000 a day</span>
          </p>
        </div>
      ))}
    </div>
  )
}
