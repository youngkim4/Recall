// The filters of a search: what they can be, and what a search asks the
// server for with them. The server's words for each are in
// recall_services/README.md (GET /api/search).

import type { Choice } from './choice'
import type { PickerOption } from './scope'

export type SearchFilters = {
  // one conversation, or '' for all
  contact: string
  from: '' | 'me' | 'them'
  kind: '' | 'direct' | 'group'
  // '' for anything; 'any' for some attachment, 'none' for none
  attachments: '' | 'any' | 'none' | 'image' | 'video' | 'audio' | 'other'
  // "YYYY-MM-DD", or '' for no limit on that side
  since: string
  until: string
  order: 'newest' | 'oldest' | 'relevance'
}

export const NO_FILTERS: SearchFilters = {
  contact: '',
  from: '',
  kind: '',
  attachments: '',
  since: '',
  until: '',
  order: 'newest',
}

export const FROM_CHOICES: Choice[] = [
  { value: '', label: 'Anyone' },
  { value: 'me', label: 'Me' },
  { value: 'them', label: 'Others' },
]

// A kind of chat is chosen in the same picker as one conversation, under
// "All conversations": both say where to search, and one picker keeps the row short.
const KIND = 'kind:'

export function whereChoices(scopes: PickerOption[]): PickerOption[] {
  const [everywhere, ...conversations] = scopes
  return [
    everywhere,
    { value: `${KIND}direct`, label: 'One-to-one chats' },
    { value: `${KIND}group`, label: 'Group chats' },
    ...conversations,
  ]
}

// the picker's value for these filters
export function whereOf(filters: SearchFilters) {
  return filters.contact || (filters.kind ? `${KIND}${filters.kind}` : '')
}

// the filters with where to search set from the picker; conversation and kind never both
export function withWhere(filters: SearchFilters, value: string): SearchFilters {
  const kind = value.startsWith(KIND) ? (value.slice(KIND.length) as SearchFilters['kind']) : ''
  return { ...filters, contact: kind ? '' : value, kind }
}

export const HAS_CHOICES: Choice[] = [
  { value: '', label: 'Anything' },
  { value: 'any', label: 'An attachment' },
  { value: 'image', label: 'A photo' },
  { value: 'video', label: 'A video' },
  { value: 'audio', label: 'Audio' },
  { value: 'other', label: 'Another file' },
  { value: 'none', label: 'No attachment' },
]

export const ORDER_CHOICES: Choice[] = [
  { value: 'newest', label: 'Newest' },
  { value: 'oldest', label: 'Oldest' },
  { value: 'relevance', label: 'Best match' },
]

// whether anything narrows the search; the order only sorts it
export function filtersSet(filters: SearchFilters) {
  return Object.entries(filters).some(([name, value]) => name !== 'order' && Boolean(value))
}

// best match needs words to match; with none, the newest come first
function orderOf(filters: SearchFilters, hasWords: boolean) {
  return filters.order === 'relevance' && !hasWords ? 'newest' : filters.order
}

// What the request carries: only what is set, and no order where it is the
// server's own (newest first).
export function filterParams(filters: SearchFilters, hasWords: boolean): Partial<SearchFilters> {
  const order = orderOf(filters, hasWords)
  const set = Object.entries({ ...filters, order: order === 'newest' ? '' : order }).filter(([, value]) => value)
  return Object.fromEntries(set) as Partial<SearchFilters>
}

// tells one search's filters from another's
export function filtersKey(filters: SearchFilters) {
  return [filters.contact, filters.from, filters.kind, filters.attachments, filters.since, filters.until, filters.order].join('\n')
}

// which part of the matches a list of the first ones holds: "the newest 80 shown"
export function shownWord(filters: SearchFilters, hasWords: boolean) {
  const order = orderOf(filters, hasWords)
  return order === 'relevance' ? 'best' : order
}
