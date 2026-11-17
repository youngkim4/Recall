// The figures of a conversation as the server last sent them, kept in this
// browser so a conversation opens with something on the page. Keeping them is
// a convenience: where the store is blocked, full or damaged, everything here
// answers as if nothing had been kept.

import type { PreviewPayload } from '../types'

// the version changes with the shape of a preview, so an old shape is never read back
const CACHE_PREFIX = 'recall.previewCache.'
const PREVIEW_CACHE_KEY = `${CACHE_PREFIX}v8`
const PREVIEW_CACHE_LIMIT = 96
const PREVIEW_CACHE_TTL_MS = 1000 * 60 * 60 * 24 * 14
// the last moment a Date can hold
const LATEST_TIME = 8.64e15

type PreviewCacheEntry = {
  payload: PreviewPayload
  savedAt: number
}

type Kept = Array<[key: string, entry: PreviewCacheEntry]>

// The browser's store, or null where there is none. A page that may not keep
// anything is refused at the moment it asks for the store.
function storage(): Storage | null {
  try {
    return typeof window === 'undefined' ? null : (window.localStorage ?? null)
  } catch {
    return null
  }
}

// A preview is only used when it has the shape this build reads: one stored
// by an older build, or sent by an older server, is treated as missing.
export function isPreviewPayload(value: unknown): value is PreviewPayload {
  const candidate = value as Partial<PreviewPayload> | null
  return Boolean(
    candidate &&
      typeof candidate === 'object' &&
      candidate.stats &&
      Array.isArray(candidate.activity?.points) &&
      Array.isArray(candidate.daily?.counts) &&
      Array.isArray(candidate.recentMessages),
  )
}

function isEntry(value: unknown): value is PreviewCacheEntry {
  const entry = value as Partial<PreviewCacheEntry> | null
  return Boolean(
    entry &&
      typeof entry === 'object' &&
      typeof entry.savedAt === 'number' &&
      Math.abs(entry.savedAt) <= LATEST_TIME &&
      isPreviewPayload(entry.payload),
  )
}

// What a stored value still holds at `now`: the entries that are whole and no
// more than two weeks old, newest first, up to the limit. A preview is asked
// for again each time it is shown, so when it was kept is also when it was
// last of use.
function keptIn(stored: unknown, now: number): Kept {
  if (!stored || typeof stored !== 'object' || Array.isArray(stored)) return []
  const whole = Object.entries(stored).filter((pair): pair is Kept[number] => isEntry(pair[1]))
  return whole
    .filter(([, entry]) => now - entry.savedAt <= PREVIEW_CACHE_TTL_MS)
    .sort(([, a], [, b]) => b.savedAt - a.savedAt)
    .slice(0, PREVIEW_CACHE_LIMIT)
}

function readKept(now: number): Kept {
  try {
    const raw = storage()?.getItem(PREVIEW_CACHE_KEY)
    return raw ? keptIn(JSON.parse(raw), now) : []
  } catch {
    return []
  }
}

// previews kept in an older shape are of no use; their space is given back
function dropOlderShapes(from: Storage) {
  Array.from({ length: from.length }, (_, index) => from.key(index))
    .filter((key): key is string => Boolean(key?.startsWith(CACHE_PREFIX)) && key !== PREVIEW_CACHE_KEY)
    .forEach((key) => from.removeItem(key))
}

// A store with no room for all of them takes the newer half, and half of that, until they fit.
function save(into: Storage, kept: Kept) {
  try {
    into.setItem(PREVIEW_CACHE_KEY, JSON.stringify(Object.fromEntries(kept)))
  } catch {
    if (kept.length > 1) save(into, kept.slice(0, Math.ceil(kept.length / 2)))
  }
}

function writeKept(kept: Kept) {
  const into = storage()
  if (!into) return
  try {
    dropOlderShapes(into)
  } catch {
    // what is left behind only takes up room
  }
  save(into, kept)
}

export function makePreviewCacheKey(input: {
  messagesPath: string
  contact: string
  model: string
  since?: string
  until?: string
}) {
  return [
    input.messagesPath,
    input.contact,
    input.model,
    input.since || '',
    input.until || '',
  ].join('::')
}

export function readPreviewCache(key: string, now = Date.now()) {
  if (!key) return null
  return readKept(now).find(([name]) => name === key)?.[1].payload ?? null
}

export function writePreviewCache(key: string, payload: PreviewPayload, now = Date.now()) {
  if (!key) return
  const newest: Kept[number] = [key, { payload, savedAt: now }]
  const others = readKept(now).filter(([name]) => name !== key)
  writeKept([newest, ...others].slice(0, PREVIEW_CACHE_LIMIT))
}

export function previewCacheStats(now = Date.now()) {
  const kept = readKept(now)
  return {
    count: kept.length,
    lastSavedAt: kept.length ? new Date(kept[0][1].savedAt).toISOString() : '',
  }
}

// How many times the saved figures were cleared in this session. Figures
// held in memory from before a clearing are of the archive as it was, and a
// view that holds some watches this number to drop them.
let generation = 0
const watchers = new Set<() => void>()

export function previewGeneration() {
  return generation
}

// Calls `listener` after each clearing. Returns the way to stop.
export function watchPreviewCache(listener: () => void) {
  watchers.add(listener)
  return () => {
    watchers.delete(listener)
  }
}

export function clearPreviewCache() {
  try {
    storage()?.removeItem(PREVIEW_CACHE_KEY)
  } catch {
    // a store that cannot be reached holds nothing to clear
  }
  generation += 1
  watchers.forEach((listener) => listener())
}
