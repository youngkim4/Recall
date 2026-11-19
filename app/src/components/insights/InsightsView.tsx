import { useMemo, useState } from 'react'
import { axisLabels, pointName, type AxisLabel } from '../../lib/activity'
import { formatDateRange, formatNumber } from '../../lib/format'
import { groupKind, querySentence, statsAsk, statsKey, type StatsBy, type StatsRow } from '../../lib/insights'
import { loadStats } from '../../lib/insightsLoad'
import { hourName, weekdayName } from '../../lib/locale'
import { loadPatterns } from '../../lib/overviewLoads'
import { scopeChoices } from '../../lib/scope'
import { NO_FILTERS, type SearchFilters } from '../../lib/searchFilters'
import { useLoad } from '../../lib/useLoad'
import { useLastReady, useSettled } from '../../lib/useSettled'
import type { Contact, Defaults } from '../../types'
import { ColumnChart } from './ColumnChart'
import { InsightsQuery } from './InsightsQuery'
import { InsightsTable } from './InsightsTable'

type InsightsViewProps = {
  defaults: Defaults | null
  contacts: Contact[]
  onSelectContact: (contact: Contact) => void
}

// words are counted once typing has paused this long
const WORDS_WAIT_MS = 400
// people and conversations listed by who
const WHO_ROWS = 25

// The server counts weekdays from Monday ("0") and hours from midnight ("00");
// the names follow the reader's locale, which counts weekdays from Sunday.
const weekdayOf = (row: StatsRow) => (Number(row.key) + 1) % 7

function namer(by: StatsBy) {
  if (by === 'day' || by === 'week' || by === 'month') return (row: StatsRow) => pointName(row.key, by)
  if (by === 'weekday') return (row: StatsRow) => weekdayName(weekdayOf(row), 'long')
  if (by === 'hour') return (row: StatsRow) => hourName(Number(row.key))
  return (row: StatsRow) => row.label
}

function labelsOf(by: StatsBy, rows: StatsRow[]): AxisLabel[] {
  if (by === 'day' || by === 'week' || by === 'month') return axisLabels(rows, by)
  if (by === 'weekday') return rows.map((row, index) => ({ index, text: weekdayName(weekdayOf(row), 'short') }))
  // every third hour
  return rows.flatMap((row, index) => (index % 3 === 0 ? [{ index, text: hourName(Number(row.key)) }] : []))
}

const TABLE: Record<string, { heading: string; column: string }> = {
  day: { heading: 'Busiest days', column: 'Day' },
  week: { heading: 'Busiest weeks', column: 'Week' },
  month: { heading: 'Busiest months', column: 'Month' },
  weekday: { heading: 'Busiest weekdays', column: 'Weekday' },
  hour: { heading: 'Busiest hours', column: 'Hour' },
  person: { heading: 'People', column: 'Person' },
  conversation: { heading: 'Conversations', column: 'Conversation' },
}

// Insights asks one question of the whole archive at a time, after
// Amplitude's chart builder: the question on the left, part by part; on the
// right the question read back as a sentence with its total, the answer as
// columns, and a table of its busiest rows. Nothing here repeats the
// Overview, which shows the archive as it is; this answers what is asked.
export function InsightsView({ defaults, contacts, onSelectContact }: InsightsViewProps) {
  const messagesPath = defaults?.messagesPath ?? ''
  const [typed, setTyped] = useState('')
  const words = useSettled(typed, WORDS_WAIT_MS)
  const [filters, setFilters] = useState<SearchFilters>(NO_FILTERS)
  const [by, setBy] = useState<StatsBy>('month')
  const scopes = useMemo(() => scopeChoices(contacts), [contacts])
  const { load: patterns } = useLoad(messagesPath, loadPatterns)
  const asked = statsAsk(filters, words, by, groupKind(by) === 'who' ? WHO_ROWS : undefined)
  const answer = useLastReady(useLoad(statsKey(messagesPath, asked), loadStats).load)
  // the answer on the page may be of the question before this one, while this one loads
  const answeredBy = (answer?.by as StatsBy | undefined) ?? by
  const who = groupKind(answeredBy) === 'who'
  const nameOf = namer(answeredBy)
  const rows = answer?.rows ?? []

  const where = scopes.find((scope) => scope.value === filters.contact)?.label
  const range =
    filters.since || filters.until ? formatDateRange(filters.since || answer?.first, filters.until || answer?.last) : ''
  const sentence = querySentence({ words, filters, by, where, range })

  function open(row: StatsRow) {
    const id = row.conversations?.[0] ?? row.key
    onSelectContact(contacts.find((contact) => contact.chat_id === id) ?? { chat_id: id, display_name: row.label })
  }

  return (
    <div className="view">
      <div className="page wide insights arrive-each">
        <InsightsQuery
          words={typed}
          onWordsChange={setTyped}
          filters={filters}
          onFiltersChange={setFilters}
          by={by}
          onByChange={setBy}
          scopes={scopes}
          daily={patterns.state === 'ready' ? patterns.value.daily : null}
          disabled={!messagesPath}
        />

        <div className="insights-answer">
          <header className="insights-head">
            <h1 key={sentence} className="insights-question swap">
              {sentence}
            </h1>
            <p className="insights-total" aria-busy={!answer || undefined}>
              {answer ? (
                <>
                  <span className="insights-count">
                    <b>{formatNumber(answer.totals.total)}</b> {answer.totals.total === 1 ? 'message' : 'messages'}
                  </span>
                  <span className="readout-voice">
                    <i className="you" />
                    You <b>{formatNumber(answer.totals.sent)}</b>
                  </span>
                  <span className="readout-voice">
                    <i className="them" />
                    Them <b>{formatNumber(answer.totals.received)}</b>
                  </span>
                </>
              ) : (
                <span className="skeleton">000,000 messages</span>
              )}
            </p>
          </header>

          {answer && !answer.totals.total ? (
            <p className="insights-none">No messages answer this question</p>
          ) : !who && rows.length ? (
            <ColumnChart
              rows={rows}
              labels={labelsOf(answeredBy, rows)}
              nameOf={nameOf}
              summary={`${sentence}: ${formatNumber(answer?.totals.total ?? 0)} messages.`}
            />
          ) : null}

          {answer && answer.totals.total ? (
            <InsightsTable
              rows={rows}
              who={who}
              heading={TABLE[answeredBy]?.heading ?? 'Rows'}
              column={TABLE[answeredBy]?.column ?? 'Row'}
              nameOf={nameOf}
              total={answer.totals.total}
              onOpen={open}
            />
          ) : null}
        </div>
      </div>
    </div>
  )
}
