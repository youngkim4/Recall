import { formatDate, parseDate, plural } from '../../lib/format'
import { useFitText } from '../../lib/useFitText'
import type { ArchiveSummary } from '../../types'

// the sizes the masthead's line is set between, in pixels
const TITLE_MIN = 24
const TITLE_MAX = 64

const monthYear = new Intl.DateTimeFormat(undefined, { month: 'long', year: 'numeric' })
const dateline = new Intl.DateTimeFormat(undefined, { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })

// The masthead of the app, as a newspaper's front page has one: the archive
// in one line set large (how many messages, in how many conversations, since
// when), a double rule under it, and under the rule the dateline, today's
// date at the left and through which day the archive is imported at the
// right. Today's date ties the page to the day, as the moments under it are.
// (A strip of four small figures with Update at its far end read as a
// dashboard and was refused; a quiet line was not "attention grabbing" enough.
// The user picked this from three mockups. Update is in Settings.)
export function ArchiveHead({ archive }: { archive: ArchiveSummary }) {
  const first = parseDate(archive.firstMessage)
  const last = parseDate(archive.lastMessage)
  const today = new Date()
  const title = `${plural(archive.messages, 'message')} in ${plural(archive.conversations, 'conversation')}${
    first ? `, since ${monthYear.format(first)}` : ''
  }`
  // one line from edge to edge, as a masthead is set; it wraps only where 24px will not fit
  const fit = useFitText<HTMLHeadingElement>(title, TITLE_MIN, TITLE_MAX)
  return (
    <header className="overview-head">
      <h1 ref={fit} className="overview-title">
        {title}
      </h1>
      <div className="overview-rule" aria-hidden="true" />
      <p className="overview-dateline">
        <time dateTime={today.toISOString().slice(0, 10)}>{dateline.format(today)}</time>
        {last ? <span className="overview-imported">Imported through {formatDate(last)}</span> : null}
      </p>
    </header>
  )
}

// holds the place of the masthead while the archive is first read
export function ArchiveHeadSkeleton() {
  return (
    <div className="overview-head" aria-hidden="true">
      <p className="overview-title">
        <span className="skeleton">000,000 messages in 0,000 conversations, since June 2000</span>
      </p>
      <div className="overview-rule" />
      <p className="overview-dateline">
        <span className="skeleton">Wednesday, January 1, 2000</span>
      </p>
    </div>
  )
}
