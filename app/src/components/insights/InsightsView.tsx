import { useMemo, useState } from 'react'
import { rhythmOf, statsAsk, statsKey, type StatsBy } from '../../lib/insights'
import { loadStats } from '../../lib/insightsLoad'
import { loadPatterns } from '../../lib/overviewLoads'
import { scopeChoices } from '../../lib/scope'
import { NO_FILTERS, type SearchFilters } from '../../lib/searchFilters'
import { useLoad } from '../../lib/useLoad'
import { useLastReady, useSettled } from '../../lib/useSettled'
import type { Contact, Defaults } from '../../types'
import { Rhythm } from '../overview/Rhythm'
import { SearchFilterBar } from '../search/SearchFilterBar'
import { ChoiceMenu } from '../ui/ChoiceMenu'
import { SearchField } from '../ui/SearchField'
import { VoicesOverTime, VoicesSkeleton } from '../ui/VoicesOverTime'
import { InsightsFigures } from './InsightsFigures'
import { InsightsPeople, PEOPLE_ASKED, type PeopleBy } from './InsightsPeople'

type InsightsViewProps = {
  defaults: Defaults | null
  contacts: Contact[]
  onSelectContact: (contact: Contact) => void
}

type Grain = Extract<StatsBy, 'day' | 'week' | 'month'>

const GRAINS = [
  { value: 'day', label: 'Day' },
  { value: 'week', label: 'Week' },
  { value: 'month', label: 'Month' },
]

// words are counted once typing has paused this long
const WORDS_WAIT_MS = 400

// Counts over the whole archive, after Plausible: one row of filters on top
// (words to count, then where, who, what and when, as in Search), and every
// figure and chart under it follows them. Four figures; the messages over
// time, by day, week or month; then who they are with beside the hours of
// the week they fall in.
export function InsightsView({ defaults, contacts, onSelectContact }: InsightsViewProps) {
  const messagesPath = defaults?.messagesPath ?? ''
  const [typed, setTyped] = useState('')
  const words = useSettled(typed, WORDS_WAIT_MS)
  const [filters, setFilters] = useState<SearchFilters>(NO_FILTERS)
  const [grain, setGrain] = useState<Grain>('month')
  const [peopleBy, setPeopleBy] = useState<PeopleBy>('person')
  const scopes = useMemo(() => scopeChoices(contacts), [contacts])
  // the days of the archive, for the calendar of the date filter
  const { load: patterns } = useLoad(messagesPath, loadPatterns)

  const overTime = useLastReady(useLoad(statsKey(messagesPath, statsAsk(filters, words, grain)), loadStats).load)
  const people = useLastReady(
    useLoad(statsKey(messagesPath, statsAsk(filters, words, peopleBy, PEOPLE_ASKED)), loadStats).load,
  )
  const hours = useLastReady(useLoad(statsKey(messagesPath, statsAsk(filters, words, 'weekday_hour')), loadStats).load)
  const rhythm = useMemo(() => (hours ? rhythmOf(hours.rows) : null), [hours])

  function open(conversationId: string, label: string) {
    onSelectContact(
      contacts.find((contact) => contact.chat_id === conversationId) ?? {
        chat_id: conversationId,
        display_name: label,
      },
    )
  }

  const grainMenu = (
    <ChoiceMenu label="By" value={grain} choices={GRAINS} onChange={(value) => setGrain(value as Grain)} align="end" />
  )

  return (
    <div className="view">
      <div className="page wide insights arrive-each">
        <div className="insights-bar">
          <SearchField
            value={typed}
            onChange={setTyped}
            label="Words to count"
            placeholder="Count messages with these words"
            large
            disabled={!messagesPath}
          />
          <SearchFilterBar
            filters={filters}
            onChange={setFilters}
            scopes={scopes}
            daily={patterns.state === 'ready' ? patterns.value.daily : null}
          />
        </div>

        <InsightsFigures stats={overTime} people={people} peopleBy={peopleBy} />

        <section className="section ruled" aria-labelledby="insights-time">
          {overTime && overTime.rows.length ? (
            <VoicesOverTime
              heading={
                <span className="insights-chart-head">
                  <h2 id="insights-time">Over time</h2>
                  {grainMenu}
                </span>
              }
              points={overTime.rows}
              unit={grain}
              sent={overTime.totals.sent}
              received={overTime.totals.received}
              them="Everyone else"
            />
          ) : overTime ? (
            <>
              <div className="section-head">
                <h2 id="insights-time">Over time</h2>
              </div>
              <p className="section-note">No messages match these filters</p>
            </>
          ) : (
            <VoicesSkeleton heading={<h2 id="insights-time">Over time</h2>} />
          )}
        </section>

        <div className="overview-row">
          {/* a column, so the list takes the height the hours give the row (overview.css) */}
          <div className="overview-col">
            <InsightsPeople by={peopleBy} onByChange={setPeopleBy} stats={people} onOpen={open} />
          </div>
          {rhythm ? <Rhythm rhythm={rhythm} /> : <section className="section ruled" aria-hidden="true" />}
        </div>
      </div>
    </div>
  )
}
