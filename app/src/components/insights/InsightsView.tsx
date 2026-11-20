import { useMemo, useState } from 'react'
import { axisLabels, pointName, type AxisLabel } from '../../lib/activity'
import type { Choice } from '../../lib/choice'
import { formatNumber } from '../../lib/format'
import { groupKind, querySentence, statsAsk, statsKey, type StatsBy, type StatsRow } from '../../lib/insights'
import { loadStats } from '../../lib/insightsLoad'
import { hourName, weekdayName } from '../../lib/locale'
import { loadPatterns } from '../../lib/overviewLoads'
import { scopeChoices } from '../../lib/scope'
import { NO_FILTERS, type SearchFilters } from '../../lib/searchFilters'
import { useLoad } from '../../lib/useLoad'
import { useLastReady, useSettled } from '../../lib/useSettled'
import type { Contact, Defaults } from '../../types'
import { SearchFilterBar } from '../search/SearchFilterBar'
import { ChoiceMenu } from '../ui/ChoiceMenu'
import { SearchField } from '../ui/SearchField'
import { ColumnChart } from './ColumnChart'
import { InsightsHabits } from './InsightsHabits'
import { InsightsTable } from './InsightsTable'
import { InsightsTabs } from './InsightsTabs'
import { LineChart, type LineSeries } from './LineChart'
import { WhenGrid } from './WhenGrid'

type InsightsViewProps = {
  defaults: Defaults | null
  contacts: Contact[]
  onSelectContact: (contact: Contact) => void
}

type Grain = Extract<StatsBy, 'day' | 'week' | 'month' | 'weekday' | 'hour'>
type Who = Extract<StatsBy, 'person' | 'conversation'>

const GRAINS: Choice[] = [
  { value: 'day', label: 'Day' },
  { value: 'week', label: 'Week' },
  { value: 'month', label: 'Month' },
  { value: 'weekday', label: 'Weekday' },
  { value: 'hour', label: 'Hour' },
]
const WHO: Choice[] = [
  { value: 'person', label: 'People' },
  { value: 'conversation', label: 'Conversations' },
]

// words are counted once typing has paused this long
const WORDS_WAIT_MS = 400
// the people or conversations the table lists
const WHO_ROWS = 10

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

// Insights counts the whole archive under one question, after ElevenLabs'
// usage page and Mixpanel's breakdown: one row of controls (the words to
// count, where, from whom, with what, when, and at its end how the chart
// groups them); under it one card, its figures as tabs over the chart; under
// that the habits of the answer, when in the week it is sent, and a table of
// the people it is with and their habits. Each part follows the row.
export function InsightsView({ defaults, contacts, onSelectContact }: InsightsViewProps) {
  const messagesPath = defaults?.messagesPath ?? ''
  const [typed, setTyped] = useState('')
  const words = useSettled(typed, WORDS_WAIT_MS)
  const [filters, setFilters] = useState<SearchFilters>(NO_FILTERS)
  const [grain, setGrain] = useState<Grain>('month')
  const [who, setWho] = useState<Who>('person')
  const [series, setSeries] = useState<LineSeries>('both')
  const scopes = useMemo(() => scopeChoices(contacts), [contacts])
  const { load: patterns } = useLoad(messagesPath, loadPatterns)

  const answer = useLastReady(useLoad(statsKey(messagesPath, statsAsk(filters, words, grain)), loadStats).load)
  const counted = useLastReady(useLoad(statsKey(messagesPath, statsAsk(filters, words, who, WHO_ROWS)), loadStats).load)
  // The same people with the habits of the answer and of each of them. They
  // read every message of the answer (a second on the whole archive), so the
  // people are asked for twice: the table stands at once, its habits after.
  const { load: habitsLoad } = useLoad(statsKey(messagesPath, statsAsk(filters, words, who, WHO_ROWS, true)), loadStats)
  const people = habitsLoad.state === 'ready' ? habitsLoad.value : counted
  const week = useLastReady(useLoad(statsKey(messagesPath, statsAsk(filters, words, 'weekday_hour')), loadStats).load)
  // the words, the sender and the attachments narrow the messages, not the talks
  const narrowed = Boolean(words.trim() || filters.from || filters.attachments)
  // the answer on the page may be of the question before this one, while this one loads
  const shownGrain = (answer?.by as Grain | undefined) ?? grain
  const shownWho = (people?.by as Who | undefined) ?? who
  const nameOf = namer(shownGrain)
  const rows = answer?.rows ?? []
  const summary = `${querySentence({ words, filters, by: shownGrain })}: ${formatNumber(answer?.totals.total ?? 0)} messages.`

  function open(row: StatsRow) {
    const id = row.conversations?.[0] ?? row.key
    onSelectContact(contacts.find((contact) => contact.chat_id === id) ?? { chat_id: id, display_name: row.label })
  }

  return (
    <div className="view">
      <div className="page wide insights arrive-each">
        <div className="insights-toolbar">
          <SearchField
            value={typed}
            onChange={setTyped}
            label="Words to count"
            placeholder="Count messages with these words"
            disabled={!messagesPath}
          />
          <SearchFilterBar
            filters={filters}
            onChange={setFilters}
            scopes={scopes}
            daily={patterns.state === 'ready' ? patterns.value.daily : null}
          />
          <span className="insights-toolbar-end">
            <ChoiceMenu
              label="By"
              value={grain}
              choices={GRAINS}
              onChange={(value) => setGrain(value as Grain)}
              align="end"
            />
          </span>
        </div>

        <section className="card insights-card" aria-label="The answer">
          <InsightsTabs stats={answer} series={series} onSeriesChange={setSeries} />
          <div className="insights-plot">
            {answer && !answer.totals.total ? (
              <p className="insights-none">No messages match</p>
            ) : !rows.length ? (
              <div className="insights-plot-waiting" aria-hidden="true" />
            ) : groupKind(shownGrain) === 'time' ? (
              <LineChart
                rows={rows}
                series={series}
                labels={labelsOf(shownGrain, rows)}
                nameOf={nameOf}
                summary={summary}
              />
            ) : (
              <ColumnChart rows={rows} labels={labelsOf(shownGrain, rows)} nameOf={nameOf} summary={summary} />
            )}
          </div>
        </section>

        {answer && !answer.totals.total ? null : (
          <>
            <InsightsHabits
              habits={people?.habits ?? null}
              failed={habitsLoad.state === 'failed'}
              narrowed={narrowed}
            />
            <WhenGrid rows={week?.rows ?? null} />
          </>
        )}

        {people && people.totals.total ? (
          <InsightsTable
            rows={people.rows}
            heading={shownWho === 'person' ? 'By person' : 'By conversation'}
            column={shownWho === 'person' ? 'Person' : 'Conversation'}
            onOpen={open}
            habitsFailed={habitsLoad.state === 'failed'}
            action={
              <ChoiceMenu
                label="Rows"
                quietLabel
                value={who}
                choices={WHO}
                onChange={(value) => setWho(value as Who)}
                align="end"
              />
            }
          />
        ) : null}
      </div>
    </div>
  )
}
