import type { PreviewPayload } from '../../types'
import { People } from '../ui/People'
import { VoicesOverTime } from '../ui/VoicesOverTime'
import { conversationFacts } from './facts'

type FiguresProps = {
  figures: PreviewPayload
  // what the conversation goes by
  title: string
  group: boolean
}

// What the numbers say about a conversation over a stretch of time: the two
// voices over that time, a few facts worth a line each, and in a group, who
// wrote how much. The same wherever a conversation's figures are shown.
export function Figures({ figures, title, group }: FiguresProps) {
  const { stats, activity } = figures
  return (
    <>
      <section className="section" aria-label="Who wrote how much">
        <VoicesOverTime
          // a new stretch of time starts with nothing picked out
          key={`${activity.unit} ${activity.points[0]?.key} ${activity.points.length}`}
          points={activity.points}
          unit={activity.unit}
          sent={stats.sentCount ?? 0}
          received={stats.receivedCount ?? 0}
          them={group ? 'Everyone else' : title}
        />
      </section>

      <dl className="section facts">
        {conversationFacts(stats, figures.dynamics, group ? 'Someone else' : title).map((fact) => (
          <div key={fact.label}>
            <dt>{fact.label}</dt>
            <dd>{fact.value}</dd>
          </div>
        ))}
      </dl>

      {group ? <People preview={figures} /> : null}
    </>
  )
}
