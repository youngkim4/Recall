import type { PreviewPayload } from '../types'

const PREVIEW_CACHE_KEY = 'recall.previewCache.v1'
const PREVIEW_CACHE_LIMIT = 96
const PREVIEW_CACHE_TTL_MS = 1000 * 60 * 60 * 24 * 14

type PreviewCacheEntry = {
  payload: PreviewPayload
  savedAt: number
  lastAccessedAt: number
}

type PreviewCacheStore = Record<string, PreviewCacheEntry>

function canUseStorage() {
  return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined'
}

function readStore(): PreviewCacheStore {
  if (!canUseStorage()) return {}
  try {
    const raw = window.localStorage.getItem(PREVIEW_CACHE_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw) as PreviewCacheStore
    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch {
    return {}
  }
}

function writeStore(store: PreviewCacheStore) {
  if (!canUseStorage()) return
  try {
    window.localStorage.setItem(PREVIEW_CACHE_KEY, JSON.stringify(store))
  } catch {
    // Browser storage is best-effort; preview requests still work without it.
  }
}

function trimStore(store: PreviewCacheStore, now: number) {
  const entries = Object.entries(store)
    .filter(([, entry]) => now - entry.savedAt <= PREVIEW_CACHE_TTL_MS)
    .sort(([, a], [, b]) => b.lastAccessedAt - a.lastAccessedAt)
    .slice(0, PREVIEW_CACHE_LIMIT)

  return Object.fromEntries(entries)
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
  const store = readStore()
  const entry = store[key]
  if (!entry || now - entry.savedAt > PREVIEW_CACHE_TTL_MS) {
    if (entry) {
      delete store[key]
      writeStore(store)
    }
    return null
  }

  entry.lastAccessedAt = now
  writeStore(trimStore(store, now))
  return entry.payload
}

export function writePreviewCache(key: string, payload: PreviewPayload, now = Date.now()) {
  if (!key) return
  const store = readStore()
  store[key] = { payload, savedAt: now, lastAccessedAt: now }
  writeStore(trimStore(store, now))
}

export function previewCacheStats(now = Date.now()) {
  const store = trimStore(readStore(), now)
  writeStore(store)
  const entries = Object.values(store)
  const lastSavedAt = entries.reduce((latest, entry) => Math.max(latest, entry.savedAt), 0)
  return {
    count: entries.length,
    lastSavedAt: lastSavedAt ? new Date(lastSavedAt).toISOString() : '',
  }
}

export function clearPreviewCache() {
  if (!canUseStorage()) return
  window.localStorage.removeItem(PREVIEW_CACHE_KEY)
}
