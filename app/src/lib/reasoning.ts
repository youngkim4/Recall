// How hard a model thinks about a chat question. The server says which
// levels each model takes, least first, and which one a question gets when
// it asks for none. The level a person picks is kept for that model alone:
// each model has its own.
import type { ModelChoice } from '../types'

export type ReasoningStep = {
  // the word that is sent to the server
  value: string
  // the word a person reads
  label: string
}

// model id -> the level kept for it
export type Efforts = Readonly<Record<string, string>>

const KEY = 'recall.reasoning.v1'
const NAMES: Record<string, string> = {
  none: 'Off',
  low: 'Low',
  medium: 'Medium',
  high: 'High',
  xhigh: 'Extra high',
  max: 'Max',
}
// a level is one short plain word, as the provider writes it
const LEVEL = /^[a-z0-9_]{1,24}$/
const MOST_ID = 120
const NONE: Efforts = Object.freeze({})

function nameOf(level: string) {
  if (NAMES[level]) return NAMES[level]
  const words = level.replaceAll('_', ' ')
  return words.charAt(0).toUpperCase() + words.slice(1)
}

function levelsOf(choice: ModelChoice | undefined): string[] {
  const levels = choice?.reasoningLevels
  return Array.isArray(levels) ? levels.filter((level) => typeof level === 'string' && LEVEL.test(level)) : []
}

// The levels of a model as the steps of a control. A model the server gives
// no levels for has none, and no control.
export function reasoningSteps(choice: ModelChoice | undefined): ReasoningStep[] {
  return levelsOf(choice).map((level) => ({ value: level, label: nameOf(level) }))
}

// The level a question to this model asks for: the one kept for the model
// while the model still takes it, or the model's own default. A model with
// levels and no default among them gets the level in the middle, so that
// what a control shows is always what is sent. '' is for a model with no
// levels: nothing is asked for, and the server decides.
export function effortOf(choice: ModelChoice | undefined, kept: Efforts): string {
  const levels = levelsOf(choice)
  if (!choice || !levels.length) return ''
  const mine = kept[choice.id]
  if (mine && levels.includes(mine)) return mine
  const usual = choice.reasoningDefault
  return usual && levels.includes(usual) ? usual : levels[Math.floor((levels.length - 1) / 2)]
}

function parse(raw: string | null): Efforts {
  if (!raw) return NONE
  try {
    const held: unknown = JSON.parse(raw)
    if (!held || typeof held !== 'object' || Array.isArray(held)) return NONE
    const good = Object.entries(held).filter(
      ([model, level]) => model.length > 0 && model.length <= MOST_ID && typeof level === 'string' && LEVEL.test(level),
    )
    return good.length ? Object.freeze(Object.fromEntries(good)) : NONE
  } catch {
    return NONE
  }
}

// the browser's store, or null where there is none to reach
function store(): Storage | null {
  try {
    return typeof window === 'undefined' ? null : window.localStorage
  } catch {
    return null
  }
}

function rawKept(from: Storage | null): string | null {
  try {
    return from ? from.getItem(KEY) : null
  } catch {
    return null
  }
}

// what was last read, so that the same text gives the same record
let last: { raw: string | null; efforts: Efforts } = { raw: null, efforts: NONE }
// Levels that the store would not take, and the store that refused them.
// They hold for as long as the page is open, so the control still works in
// a window that keeps nothing.
let unkept: { from: Storage | null; efforts: Efforts } | null = null
const listeners = new Set<() => void>()

// The levels kept for the models. The record is the same one until a level changes.
export function readEfforts(): Efforts {
  const from = store()
  if (unkept?.from === from) return unkept.efforts
  const raw = rawKept(from)
  if (raw !== last.raw) last = { raw, efforts: parse(raw) }
  return last.efforts
}

// Keeps a level for one model and tells whoever listens. A word that is no
// level, and a level that is already kept, change nothing.
export function storeEffort(model: string, level: string) {
  if (!model || model.length > MOST_ID || !LEVEL.test(level)) return
  const held = readEfforts()
  if (held[model] === level) return
  const next: Efforts = Object.freeze({ ...held, [model]: level })
  const from = store()
  try {
    if (!from) throw new Error('no store')
    from.setItem(KEY, JSON.stringify(next))
    unkept = null
  } catch {
    // nowhere to keep it: it holds while the page is open
    unkept = { from, efforts: next }
  }
  for (const listener of listeners) listener()
}

// a level set in another window of the app is read here too
function onStorage(event: StorageEvent) {
  if (event.key !== null && event.key !== KEY) return
  for (const listener of listeners) listener()
}

export function subscribeEfforts(listener: () => void): () => void {
  if (!listeners.size && typeof window !== 'undefined') window.addEventListener?.('storage', onStorage)
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
    if (!listeners.size && typeof window !== 'undefined') window.removeEventListener?.('storage', onStorage)
  }
}
