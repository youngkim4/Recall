import { useState } from 'react'
import { pointName, type ActivityPoint, type ActivityUnit } from '../../lib/activity'
import { formatNumber } from '../../lib/format'
import { ActivityChart } from './ActivityChart'
import { Truncate } from './Truncate'
import { VoiceBar } from './VoiceBar'

type VoicesOverTimeProps = {
  // one bar a day or a month, oldest first
  points: ActivityPoint[]
  unit: ActivityUnit
  // what the two voices come to over the whole stretch
  sent: number
  received: number
  // who the gray voice is: a name, or "Everyone else"
  them: string
}

function summaryOf(points: ActivityPoint[], unit: ActivityUnit) {
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
export function VoicesOverTime({ points, unit, sent, received, them }: VoicesOverTimeProps) {
  const [index, setIndex] = useState<number | null>(null)
  const picked = index === null ? null : (points[index] ?? null)
  const shownSent = picked ? picked.sent : sent
  const shownReceived = picked ? picked.received : received

  return (
    <div className="voices">
      <p className="readout">
        <span className="readout-voice">
          <i className="you" />
          You <b>{formatNumber(shownSent)}</b>
        </span>
        <span className="readout-voice">
          <i className="them" />
          <Truncate className="readout-name">{them}</Truncate> <b>{formatNumber(shownReceived)}</b>
        </span>
        <span className="readout-when">{picked ? pointName(picked.key, unit) : null}</span>
      </p>
      <VoiceBar sent={shownSent} received={shownReceived} />
      <ActivityChart points={points} unit={unit} active={index} onActiveChange={setIndex} summary={summaryOf(points, unit)} />
      <p className="sr-only" aria-live="polite">
        {picked
          ? `${pointName(picked.key, unit)}: you ${formatNumber(picked.sent)}, ${them} ${formatNumber(picked.received)}`
          : ''}
      </p>
    </div>
  )
}

// holds the place of the figures and the chart while they are worked out
export function VoicesSkeleton() {
  return (
    <div className="voices" aria-hidden="true">
      <p className="readout">
        <span>
          <span className="skeleton">You 00,000</span>
        </span>
        <span>
          <span className="skeleton">Their name 00,000</span>
        </span>
      </p>
      <div className="skeleton voices-skeleton-bar" />
      <div className="skeleton voices-skeleton-chart" />
    </div>
  )
}
