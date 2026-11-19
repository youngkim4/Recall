import { useState, type ReactNode } from 'react'
import { pointName, type ActivityPoint, type ChartUnit } from '../../lib/activity'
import { formatNumber } from '../../lib/format'
import { ActivityChart } from './ActivityChart'
import { Truncate } from './Truncate'
import { VoiceBar } from './VoiceBar'

type VoicesOverTimeProps = {
  // one bar a day or a month, oldest first
  points: ActivityPoint[]
  unit: ChartUnit
  // what the two voices come to over the whole stretch
  sent: number
  received: number
  // who the gray voice is: a name, or "Everyone else"
  them: string
  // The heading of the section the chart stands in, where it is an open one.
  // The figures then share the heading's line, at the end of it, the way the
  // reading of a heat map does, and the bar is the first thing under it.
  heading?: ReactNode
}

type FrameProps = {
  heading?: ReactNode
  // the two totals
  figures: ReactNode
  // the day or month that is picked out, where one can be
  when?: ReactNode
  // it only holds a place: nothing in it is read out
  waiting?: boolean
  children: ReactNode
}

// Where the figures stand. With a heading they end the heading's line, and
// the day or month picked out stands between the two, in the room that is
// left: it is no part of the figures there, so it cannot push them to
// another line. With no heading the figures are over the bar, and what is
// picked out ends their line.
function Frame({ heading, figures, when, waiting, children }: FrameProps) {
  const hidden = waiting ? true : undefined
  if (heading) {
    return (
      <>
        <div className="section-head" aria-hidden={hidden}>
          {heading}
          {when}
          <p className="readout">{figures}</p>
        </div>
        <div className="voices" aria-hidden={hidden}>
          {children}
        </div>
      </>
    )
  }
  return (
    <div className="voices" aria-hidden={hidden}>
      <p className="readout">
        {figures}
        {when}
      </p>
      {children}
    </div>
  )
}

function summaryOf(points: ActivityPoint[], unit: ChartUnit) {
  if (!points.length) return ''
  const busiest = points.reduce((best, current) => (current.total > best.total ? current : best), points[0])
  return `Messages by ${unit} from ${pointName(points[0].key, unit)} to ${pointName(
    points[points.length - 1].key,
    unit,
  )}. The busiest was ${pointName(busiest.key, unit)} with ${formatNumber(busiest.total)}.`
}

// Who wrote how much, and when: the two totals, one bar split between them,
// and the same two voices over time. Pick out a day or a month on the chart
// and the figures and the bar become that day's or month's.
export function VoicesOverTime({ points, unit, sent, received, them, heading }: VoicesOverTimeProps) {
  const [index, setIndex] = useState<number | null>(null)
  const picked = index === null ? null : (points[index] ?? null)
  const shownSent = picked ? picked.sent : sent
  const shownReceived = picked ? picked.received : received

  const figures = (
    <>
      <span className="readout-voice">
        <i className="you" />
        You <b>{formatNumber(shownSent)}</b>
      </span>
      <span className="readout-voice">
        <i className="them" />
        <Truncate className="readout-name">{them}</Truncate> <b>{formatNumber(shownReceived)}</b>
      </span>
    </>
  )
  const when = <span className="readout-when">{picked ? pointName(picked.key, unit) : null}</span>

  return (
    <Frame heading={heading} figures={figures} when={when}>
      <VoiceBar sent={shownSent} received={shownReceived} />
      <ActivityChart points={points} unit={unit} active={index} onActiveChange={setIndex} summary={summaryOf(points, unit)} />
      <p className="sr-only" aria-live="polite">
        {picked
          ? `${pointName(picked.key, unit)}: you ${formatNumber(picked.sent)}, ${them} ${formatNumber(picked.received)}`
          : ''}
      </p>
    </Frame>
  )
}

// holds the place of the figures and the chart while they are worked out,
// with the figures in the heading's line where the chart will have them there
export function VoicesSkeleton({ heading }: { heading?: ReactNode }) {
  const figures = (
    <>
      <span>
        <span className="skeleton">You 00,000</span>
      </span>
      <span>
        <span className="skeleton">Their name 00,000</span>
      </span>
    </>
  )

  return (
    <Frame heading={heading} figures={figures} waiting>
      <div className="skeleton voices-skeleton-bar" />
      <div className="skeleton voices-skeleton-chart" />
    </Frame>
  )
}
