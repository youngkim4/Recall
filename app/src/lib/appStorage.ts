// What the app remembers in this browser between launches: the quality level,
// where its files are, and how its panes are sized. All of it is a
// convenience; the app runs the same without it.

import type { Defaults, RuntimePaths } from '../types'
import { LIST_WIDTH } from './layout.ts'

const MODEL_KEY = 'recall.selectedModel'
const PATHS_KEY = 'recall.runtimePaths.v1'
const LAYOUT_KEY = 'recall.layout.v2'

export const RAIL_WIDTH = { min: 220, max: 340, initial: 236 }
const REPORT_LIST_INITIAL = 318

export type LayoutState = {
  sidebarWidth: number
  listWidth: number
  reportListWidth: number
  sidebarCollapsed: boolean
}

export const emptyPaths: RuntimePaths = { messagesPath: '', outDir: '' }

const defaultLayout: LayoutState = {
  sidebarWidth: RAIL_WIDTH.initial,
  listWidth: LIST_WIDTH.initial,
  reportListWidth: REPORT_LIST_INITIAL,
  sidebarCollapsed: false,
}

function read<Value>(key: string): Partial<Value> | null {
  try {
    const stored = window.localStorage.getItem(key)
    const parsed: unknown = stored ? JSON.parse(stored) : null
    return parsed && typeof parsed === 'object' ? (parsed as Partial<Value>) : null
  } catch {
    return null
  }
}

function write(key: string, value: unknown) {
  try {
    window.localStorage.setItem(key, typeof value === 'string' ? value : JSON.stringify(value))
  } catch {
    // remembering is optional
  }
}

export function preferredModel(models: string[], fallback: string) {
  try {
    const stored = window.localStorage.getItem(MODEL_KEY)
    return stored && models.includes(stored) ? stored : fallback
  } catch {
    return fallback
  }
}

export function storePreferredModel(model: string) {
  write(MODEL_KEY, model)
}

export function pathsFromDefaults(defaults: Defaults): RuntimePaths {
  return { messagesPath: defaults.messagesPath, outDir: defaults.outDir }
}

export function hasRuntimePath(paths: RuntimePaths) {
  return Boolean(paths.messagesPath || paths.outDir)
}

// Only the two locations are read. A path to a database that an earlier
// build remembered is left behind: an update reads Messages itself.
export function readStoredPaths(fallback: RuntimePaths): RuntimePaths {
  const stored = read<RuntimePaths>(PATHS_KEY)
  if (!stored) return fallback
  return {
    messagesPath: stored.messagesPath || fallback.messagesPath,
    outDir: stored.outDir || fallback.outDir,
  }
}

export function storeRuntimePaths(paths: RuntimePaths) {
  write(PATHS_KEY, paths)
}

// A remembered size, held to what the pane can be dragged to. Anything that
// is not a number is no size at all, and the pane takes its first one.
function within(value: unknown, min: number, max: number, fallback: number) {
  return typeof value === 'number' && Number.isFinite(value) ? Math.max(min, Math.min(max, value)) : fallback
}

export function readStoredLayout(): LayoutState {
  const stored = read<LayoutState>(LAYOUT_KEY)
  if (!stored) return defaultLayout
  return {
    sidebarWidth: within(stored.sidebarWidth, RAIL_WIDTH.min, RAIL_WIDTH.max, defaultLayout.sidebarWidth),
    listWidth: within(stored.listWidth, LIST_WIDTH.min, LIST_WIDTH.max, defaultLayout.listWidth),
    reportListWidth: within(stored.reportListWidth, LIST_WIDTH.min, LIST_WIDTH.max, defaultLayout.reportListWidth),
    sidebarCollapsed: stored.sidebarCollapsed === true,
  }
}

export function storeLayout(layout: LayoutState) {
  write(LAYOUT_KEY, layout)
}
