import { useEffect, useId, useMemo, useState } from 'react'
import { formatNumber } from '../../lib/format'
import { dayOfMoment } from '../../lib/momentDay'
import { leadingMoments, momentsByYear, readMoments, type Moment } from '../../lib/reportMoments'
import type { AnalysisEvent } from '../../types'
import { Thread } from '../ui/Thread'
import { bringIntoView } from './bringIntoView'
import { MomentDate } from './MomentDate'
import { MomentStrip } from './MomentStrip'

type ReportMomentsProps = {
  events: AnalysisEvent[]
  // the first and last day the report covers, "YYYY-MM-DD"
  first: string
  last: string
  // on paper nothing is folded away
  paper: boolean
}

// how many moments show before the list is opened
const LEADING = 8
// a list this little longer than that is shown whole
const SHOWN_WHOLE = LEADING + 3
const FOUND_MS = 1600
const dayFormat = new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric' })

const dayInYear = (day: string) =>
  dayFormat.format(new Date(Number(day.slice(0, 4)), Number(day.slice(5, 7)) - 1, Number(day.slice(8, 10))))

// The words a moment quotes. When the quote says who wrote each line they
// are set as the messages they were, yours in blue on the right; when it
// does not, they are a quotation and take no side.
function Said({ moment }: { moment: Moment }) {
  if (!moment.said.length) return null
  if (moment.said.some((line) => line.fromMe === null)) {
    return (
      <blockquote>
        {moment.said.map((line, index) => (
          <p key={index}>{line.text}</p>
        ))}
      </blockquote>
    )
  }
  return (
    <Thread
      messages={moment.said.map((line, index) => ({
        key: String(index),
        // no time: the day is beside the moment already
        timestamp: '',
        text: line.text,
        fromMe: line.fromMe,
        sender: line.sender || undefined,
      }))}
    />
  )
}

// The moments a report picked out of a conversation, as an open section.
// Along the top, where each one fell in time; under it, the ones that matter
// most as a timeline, oldest first, with the rest a press away.
export function ReportMoments({ events, first, last, paper }: ReportMomentsProps) {
  const baseId = useId()
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState<string | null>(null)
  // the moment that was gone to from the marks, picked out for a moment
  const [target, setTarget] = useState<string | null>(null)
  const moments = useMemo(() => readMoments(events), [events])
  const leading = useMemo(() => leadingMoments(moments, LEADING), [moments])
  const folds = moments.length > SHOWN_WHOLE
  const whole = open || paper || !folds
  const years = useMemo(
    () => momentsByYear(whole ? moments : moments.filter((moment) => leading.has(moment.key))),
    [whole, moments, leading],
  )

  useEffect(() => {
    if (!target) return
    const found = document.getElementById(`${baseId}-${target}`)
    if (found) bringIntoView(found, 'center')
    const timer = window.setTimeout(() => setTarget(null), FOUND_MS)
    return () => window.clearTimeout(timer)
  }, [target, baseId])

  if (!moments.length) return null

  function goTo(key: string) {
    // a moment that is folded away is brought out first
    if (!leading.has(key)) setOpen(true)
    setTarget(key)
  }

  return (
    <section className="section ruled" aria-labelledby={`${baseId}-title`}>
      <div className="section-head">
        <h3 id={`${baseId}-title`}>Moments</h3>
        <span className="meta">{formatNumber(moments.length)}</span>
      </div>
      <div className="report-moments">
        <MomentStrip
          moments={moments.filter((moment) => moment.day)}
          first={first}
          last={last}
          leading={leading}
          active={active}
          onActive={setActive}
          onPick={goTo}
        />
        <ol className="report-timeline" role="list">
          {years.map((year) => (
            <li key={year.year || 'undated'} className="report-timeline-year">
              {year.year ? <h4>{year.year}</h4> : null}
              <ol role="list">
                {year.moments.map((moment) => (
                  <li
                    key={moment.key}
                    id={`${baseId}-${moment.key}`}
                    className="report-moment"
                    data-lead={leading.has(moment.key) || undefined}
                    data-target={moment.key === target || undefined}
                  >
                    <p className="report-moment-when">
                      {moment.day ? (
                        // on paper a date opens nothing
                        <MomentDate opens={paper ? null : dayOfMoment(events, moment.key)}>
                          <time dateTime={moment.day}>{dayInYear(moment.day)}</time>
                        </MomentDate>
                      ) : null}
                      {moment.kind ? <span>{moment.kind}</span> : null}
                    </p>
                    <div className="report-moment-main">
                      <h5>{moment.title || 'Untitled moment'}</h5>
                      {moment.detail ? <p>{moment.detail}</p> : null}
                      <Said moment={moment} />
                    </div>
                  </li>
                ))}
              </ol>
            </li>
          ))}
        </ol>
        {folds ? (
          <p className="report-timeline-more" data-screen-only>
            <button type="button" className="link" aria-expanded={open} onClick={() => setOpen((current) => !current)}>
              {open ? `Show the ${LEADING} that matter most` : `Show all ${formatNumber(moments.length)}`}
            </button>
          </p>
        ) : null}
      </div>
    </section>
  )
}
