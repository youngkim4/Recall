import { useId, type ReactNode } from 'react'
import { formatNumber, formatPercent } from '../../lib/format'
import { reactionKinds, sharedKinds, sidesOf, type KindCount } from '../../lib/reportNumbers'
import type { AnalysisPayload } from '../../types'
import { People } from '../ui/People'
import { Truncate } from '../ui/Truncate'
import { VoiceBar } from '../ui/VoiceBar'

type ReportNumbersProps = {
  analysis: AnalysisPayload
  // who the gray voice is: a name, or "Everyone else"
  them: string
  group: boolean
}

type Sides = { you: number; them: number }

// A bar split between the two voices, with what each side comes to under its end.
function Split({ sides, them, share }: { sides: Sides; them: string; share?: boolean }) {
  const total = Math.max(1, sides.you + sides.them)
  const say = (count: number) => (share ? formatPercent(count / total) : formatNumber(count))
  return (
    <div className="report-split">
      <VoiceBar sent={sides.you} received={sides.them} />
      <p>
        <span>
          You <b>{say(sides.you)}</b>
        </span>
        <span>
          <Truncate className="report-split-name">{them}</Truncate> <b>{say(sides.them)}</b>
        </span>
      </p>
    </div>
  )
}

// each kind with a bar as long as its share of the largest
function Kinds({ kinds }: { kinds: KindCount[] }) {
  const largest = Math.max(1, ...kinds.map((kind) => kind.count))
  return (
    <ul className="report-kinds" role="list">
      {kinds.map((kind) => (
        <li key={kind.label}>
          <span>{kind.label}</span>
          <span className="report-kind-bar" aria-hidden="true">
            <i style={{ width: `${(kind.count / largest) * 100}%` }} />
          </span>
          <b>{formatNumber(kind.count)}</b>
        </li>
      ))}
    </ul>
  )
}

type BlockProps = {
  title: string
  total?: number
  // a bar and nothing under it: a short block, which a wide page places beside taller ones
  brief?: boolean
  children: ReactNode
}

function Block({ title, total, brief, children }: BlockProps) {
  const id = useId()
  return (
    <section className="report-number" data-brief={brief || undefined} aria-labelledby={id}>
      <div className="section-head">
        <h3 id={id}>{title}</h3>
        {total ? <span className="meta">{formatNumber(total)}</span> : null}
      </div>
      {children}
    </section>
  )
}

// The smaller figures, each with a picture of itself: who writes first, what
// was shared, how messages were reacted to, and in a group who wrote how much.
export function ReportNumbers({ analysis, them, group }: ReportNumbersProps) {
  const { stats, dynamics } = analysis
  const opened = dynamics?.initiationLifetime
  const shared = sharedKinds(stats.attachments)
  const sharedSides = sidesOf(stats.attachments, 'attachments')
  const reactions = reactionKinds(stats.reactions)
  const reactionSides = sidesOf(stats.reactions, 'reactions')
  const first = typeof opened === 'number' && Number.isFinite(opened) ? Math.max(0, Math.min(1, opened)) : null
  const blocks = first !== null || shared.length > 0 || reactions.length > 0
  if (!blocks && !group) return null

  return (
    <div className="section">
      {blocks ? (
        <div className="report-numbers">
          {first === null ? null : (
            <Block title="First to write each day" brief>
              <Split sides={{ you: first, them: 1 - first }} them={group ? 'Someone else' : them} share />
            </Block>
          )}
          {shared.length ? (
            <Block title="Shared" total={Number(stats.attachments?.total || 0)}>
              {sharedSides ? <Split sides={sharedSides} them={them} /> : null}
              <Kinds kinds={shared} />
            </Block>
          ) : null}
          {reactions.length ? (
            <Block title="Reactions" total={Number(stats.reactions?.total || 0)}>
              {reactionSides ? <Split sides={reactionSides} them={them} /> : null}
              <Kinds kinds={reactions} />
            </Block>
          ) : null}
        </div>
      ) : null}
      {group ? <People preview={analysis} /> : null}
    </div>
  )
}
