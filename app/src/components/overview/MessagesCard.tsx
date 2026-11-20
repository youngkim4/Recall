import { useState } from 'react'
import { axisLabels, pointName } from '../../lib/activity'
import { formatDate, formatNumber, parseDate } from '../../lib/format'
import { statsKey } from '../../lib/insights'
import { loadStats } from '../../lib/insightsLoad'
import { STRETCHES, grainOf, stretchAsk, stretchWords, type Stretch } from '../../lib/overviewMessages'
import { useLoad } from '../../lib/useLoad'
import { useLastReady } from '../../lib/useSettled'
import { useTweened } from '../../lib/useTweened'
import type { ArchiveSummary } from '../../types'
import { LineChart } from '../insights/LineChart'
import { Segmented } from '../ui/Segmented'

const monthYear = new Intl.DateTimeFormat(undefined, { month: 'long', year: 'numeric' })

type MessagesCardProps = {
  archive: ArchiveSummary
  messagesPath: string
}

// How many messages, and their line, after Mercury's balance card: the name
// of the card and through which day the archive is imported, the switch of
// stretch at its end; under them the figure set large with what it counts
// beside it; under that what you sent and what they sent as two lines. A new
// stretch does not jump in: the figure counts and the lines move to it.
export function MessagesCard({ archive, messagesPath }: MessagesCardProps) {
  const [stretch, setStretch] = useState<Stretch>('all')
  const lastDay = archive.lastMessage?.slice(0, 10) ?? ''
  const answer = useLastReady(useLoad(statsKey(messagesPath, stretchAsk(stretch, lastDay)), loadStats).load)
  // the answer on the page may be of the stretch before, while this one loads
  const shown = (answer && STRETCHES.find((item) => grainOf(item.value) === answer.by)?.value) ?? stretch
  const grain = grainOf(shown)
  const rows = answer?.rows ?? []
  const [figure] = useTweened([answer?.totals.total ?? 0])
  const first = parseDate(archive.firstMessage)
  const last = parseDate(archive.lastMessage)
  const words = stretchWords(shown, {
    conversations: archive.conversations,
    since: first ? monthYear.format(first) : '',
  })

  return (
    <section className="card overview-messages" aria-labelledby="messages-title">
      <div className="section-head">
        <h2 id="messages-title">Messages</h2>
        {last ? <span className="meta">Through {formatDate(last)}</span> : null}
        <span className="section-links">
          <Segmented label="Stretch" value={stretch} options={STRETCHES} onChange={setStretch} />
        </span>
      </div>
      <p className="overview-figure">
        <b>{answer ? formatNumber(Math.round(figure ?? 0)) : <span className="skeleton">000,000</span>}</b>
        <span>{words}</span>
      </p>
      {rows.length > 1 ? (
        <LineChart
          rows={rows}
          series="both"
          labels={axisLabels(rows, grain)}
          nameOf={(row) => pointName(row.key, grain)}
          summary={`${formatNumber(answer?.totals.total ?? 0)} messages ${words}, by ${grain}.`}
        />
      ) : (
        <div className="overview-plot-waiting" aria-hidden="true" />
      )}
    </section>
  )
}
