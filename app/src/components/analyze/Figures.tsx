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
// wrote how much. The facts and the people keep together under the chart: as
// lines in a narrow pane, and in a wide one the facts as figures set apart by
// hairlines.
export function Figures({ figures, title, group }: FiguresProps) {
  const { stats, activity } = figures
  const facts = conversationFacts(stats, figures.dynamics, group ? 'Someone else' : title)
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

      <div className="detail-facts">
        {facts.length ? (
          <dl className="section facts">
            {facts.map((fact) => (
              <div key={fact.label}>
                <dt>{fact.label}</dt>
                <dd>{fact.value}</dd>
              </div>
            ))}
          </dl>
        ) : null}

        {group ? <People preview={figures} /> : null}
      </div>
    </>
  )
}
