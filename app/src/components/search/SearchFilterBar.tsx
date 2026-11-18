import { useMemo } from 'react'
import type { Daily } from '../../lib/days'
import type { PickerOption } from '../../lib/scope'
import {
  FROM_CHOICES,
  HAS_CHOICES,
  NO_FILTERS,
  ORDER_CHOICES,
  filtersSet,
  whereChoices,
  whereOf,
  withWhere,
  type SearchFilters,
} from '../../lib/searchFilters'
import { ChoiceMenu } from '../ui/ChoiceMenu'
import { DateRange } from '../ui/DateRange'
import { ListPicker } from '../ui/ListPicker'

type SearchFilterBarProps = {
  filters: SearchFilters
  onChange: (filters: SearchFilters) => void
  // the conversations to search in, everything first
  scopes: PickerOption[]
  // how many messages each day of the archive holds, for the calendar; null while it loads
  daily: Daily | null
}

const NO_DAYS: Daily = { start: '', counts: [] }

// The filters under the search field, after Slack's search: a row of quiet
// buttons. Where to search (everywhere, a kind of chat or one conversation),
// who wrote it, what it has, and when. A filter with nothing chosen says only
// what it filters, "From"; one that is set shows its value and is filled, so
// what narrows the list shows at a glance. Reset clears them all.
export function SearchFilterBar({ filters, onChange, scopes, daily }: SearchFilterBarProps) {
  const where = useMemo(() => whereChoices(scopes), [scopes])
  const set = <Name extends keyof SearchFilters>(name: Name) =>
    (value: string) => onChange({ ...filters, [name]: value as SearchFilters[Name] })

  return (
    <div className="search-filters" role="group" aria-label="Filters">
      <span className="filter" data-set={whereOf(filters) ? '' : undefined}>
        <ListPicker
          label="Where to search"
          value={whereOf(filters)}
          options={where}
          onChange={(value) => onChange(withWhere(filters, value))}
          searchPlaceholder="Search conversations"
          emptyText="No conversations match"
        />
      </span>
      <span className="filter" data-set={filters.from ? '' : undefined}>
        <ChoiceMenu label="From" value={filters.from} choices={FROM_CHOICES} onChange={set('from')} bare />
      </span>
      <span className="filter" data-set={filters.attachments ? '' : undefined}>
        <ChoiceMenu label="Has" value={filters.attachments} choices={HAS_CHOICES} onChange={set('attachments')} bare />
      </span>
      <span className="filter" data-set={filters.since || filters.until ? '' : undefined}>
        <DateRange
          since={filters.since}
          until={filters.until}
          daily={daily ?? NO_DAYS}
          onChange={(range) => onChange({ ...filters, since: range.since, until: range.until })}
        />
      </span>
      {filtersSet(filters) ? (
        <button
          type="button"
          className="link search-filters-reset appear"
          onClick={() => onChange({ ...NO_FILTERS, order: filters.order })}
        >
          Reset
        </button>
      ) : null}
    </div>
  )
}

type SearchSortProps = {
  order: SearchFilters['order']
  onChange: (order: SearchFilters['order']) => void
  // with no words, best match means nothing, and it is not offered
  hasWords: boolean
}

// The order of the matches, at the end of the line that counts them, where
// Slack puts "Most relevant".
export function SearchSort({ order, onChange, hasWords }: SearchSortProps) {
  const choices = hasWords ? ORDER_CHOICES : ORDER_CHOICES.filter((choice) => choice.value !== 'relevance')
  return (
    <ChoiceMenu
      label="Sort"
      quietLabel
      value={hasWords || order !== 'relevance' ? order : 'newest'}
      choices={choices}
      onChange={(value) => onChange(value as SearchFilters['order'])}
      align="end"
    />
  )
}
