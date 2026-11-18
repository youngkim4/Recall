// Which answer of a search stays on screen. Search asks twice: as the words
// are typed, with the last word as the start of a word ("hi" finds "hit"),
// and when typing stops, for whole words ("hi" is "hi" only).

import type { SearchResult } from '../types'

// what was found, the search it answers, and whether its words were whole
export type Found = { key: string; whole: boolean; results: SearchResult[]; total: number }

// The answer to show, given the one shown and one that came in. Whole words
// win over the matches found while typing, unless they find nothing where
// the typed matches found something: a search left at "pizz" still shows
// "pizza". A late answer from typing never replaces whole words.
export function keepFound(shown: Found | null, next: Found): Found {
  if (!shown || shown.key !== next.key) return next
  if (shown.whole && !next.whole) return shown
  if (next.whole && !shown.whole && next.total === 0 && shown.total > 0) return shown
  return next
}
