import { useMemo } from 'react'
import type { Choice } from '../../lib/choice'
import type { Daily } from '../../lib/days'
import type { StatsBy } from '../../lib/insights'
import type { PickerOption } from '../../lib/scope'
import {
  FROM_CHOICES,
  HAS_CHOICES,
  NO_FILTERS,
  filtersSet,
  whereChoices,
  whereOf,
  withWhere,
  type SearchFilters,
} from '../../lib/searchFilters'
import { ChoiceMenu } from '../ui/ChoiceMenu'
import { DateRange } from '../ui/DateRange'
import { ListPicker } from '../ui/ListPicker'
import { SearchField } from '../ui/SearchField'

type InsightsQueryProps = {
  words: string
  onWordsChange: (words: string) => void
  filters: SearchFilters
  onFiltersChange: (filters: SearchFilters) => void
  by: StatsBy
  onByChange: (by: StatsBy) => void
  scopes: PickerOption[]
  // how many messages each day of the archive holds, for the calendar; null while it loads
  daily: Daily | null
  disabled: boolean
}

const GROUPS: Choice[] = [
  { value: 'day', label: 'Day' },
  { value: 'week', label: 'Week' },
  { value: 'month', label: 'Month' },
  { value: 'weekday', label: 'Weekday' },
  { value: 'hour', label: 'Hour' },
  { value: 'person', label: 'Person' },
  { value: 'conversation', label: 'Conversation' },
]

const NO_DAYS: Daily = { start: '', counts: [] }

// The question, after Amplitude's panel beside its chart: one labelled row a
// part of it, top to bottom, what to count, where, from whom, with what, when,
// and how to group it. A part that is set shows its value; Reset clears them.
export function InsightsQuery({
  words,
  onWordsChange,
  filters,
  onFiltersChange,
  by,
  onByChange,
  scopes,
  daily,
  disabled,
}: InsightsQueryProps) {
  const where = useMemo(() => whereChoices(scopes), [scopes])
  const set = (patch: Partial<SearchFilters>) => onFiltersChange({ ...filters, ...patch })

  return (
    <aside className="insights-query" aria-label="The question">
      <div className="query-row">
        <span className="query-label">Count</span>
        <SearchField
          value={words}
          onChange={onWordsChange}
          label="Words to count"
          placeholder="Any words"
          disabled={disabled}
        />
      </div>
      <div className="query-row" data-set={whereOf(filters) ? '' : undefined}>
        <span className="query-label">In</span>
        <ListPicker
          label="Where to count"
          value={whereOf(filters)}
          options={where}
          onChange={(value) => onFiltersChange(withWhere(filters, value))}
          searchPlaceholder="Search conversations"
          emptyText="No conversations match"
        />
      </div>
      <div className="query-row" data-set={filters.from ? '' : undefined}>
        <span className="query-label">From</span>
        <ChoiceMenu
          label="From"
          quietLabel
          value={filters.from}
          choices={FROM_CHOICES}
          onChange={(from) => set({ from: from as SearchFilters['from'] })}
        />
      </div>
      <div className="query-row" data-set={filters.attachments ? '' : undefined}>
        <span className="query-label">Has</span>
        <ChoiceMenu
          label="Has"
          quietLabel
          value={filters.attachments}
          choices={HAS_CHOICES}
          onChange={(attachments) => set({ attachments: attachments as SearchFilters['attachments'] })}
        />
      </div>
      <div className="query-row" data-set={filters.since || filters.until ? '' : undefined}>
        <span className="query-label">When</span>
        <DateRange
          since={filters.since}
          until={filters.until}
          daily={daily ?? NO_DAYS}
          onChange={(range) => set({ since: range.since, until: range.until })}
        />
      </div>
      <div className="query-row query-group">
        <span className="query-label">By</span>
        <ChoiceMenu
          label="Group by"
          quietLabel
          value={by}
          choices={GROUPS}
          onChange={(value) => onByChange(value as StatsBy)}
        />
      </div>
      {filtersSet(filters) || words ? (
        <button
          type="button"
          className="link query-reset appear"
          onClick={() => {
            onWordsChange('')
            onFiltersChange(NO_FILTERS)
          }}
        >
          Reset
        </button>
      ) : null}
    </aside>
  )
}
