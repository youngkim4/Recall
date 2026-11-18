// The last answer of each loader for each key (such as the path of an
// archive), kept in memory while the app is open. A page that opens again
// shows it at once while it asks for a fresh one, and the app can ask for a
// page's answers before the page is opened. Two asks while one request is
// out share that request. Nothing here is written to the browser's store.

type Loader<Value> = (key: string) => Promise<Value>

type Entry = {
  // the last good answer, in a box so an answer of undefined is still an answer
  answer?: { value: unknown }
  // the request that is out, if one is
  pending?: Promise<unknown>
}

let entries = new WeakMap<Loader<unknown>, Map<string, Entry>>()
// grows each time everything is forgotten, so a request that was out at that moment keeps nothing
let generation = 0

function entriesOf(loader: Loader<unknown>) {
  const known = entries.get(loader)
  if (known) return known
  const fresh = new Map<string, Entry>()
  entries.set(loader, fresh)
  return fresh
}

// the last good answer of this loader for this key, or undefined when there is none
export function keptAnswer<Value>(loader: Loader<Value>, key: string): { value: Value } | undefined {
  const answer = entries.get(loader as Loader<unknown>)?.get(key)?.answer
  return answer as { value: Value } | undefined
}

// Ask the loader for a fresh answer and keep it. While a request for the
// same loader and key is out, it is shared. A failure keeps the last good
// answer. An empty key is never asked for.
export function fetchKept<Value>(loader: Loader<Value>, key: string): Promise<Value> {
  if (!key) return Promise.reject(new Error('Nothing to load.'))
  const own = entriesOf(loader as Loader<unknown>)
  const out = own.get(key)?.pending
  if (out) return out as Promise<Value>
  const asked = generation
  const settle = (answer: Entry['answer']) => {
    if (asked !== generation) return
    const before = own.get(key)
    own.set(key, { answer: answer ?? before?.answer })
  }
  // a loader that throws before it returns a promise fails like one that rejects
  const pending = new Promise<Value>((resolve) => resolve(loader(key))).then(
    (value) => {
      settle({ value })
      return value
    },
    (error: unknown) => {
      settle(undefined)
      throw error
    },
  )
  own.set(key, { ...own.get(key), pending })
  return pending
}

// ask ahead of time, for a page that is likely to open; a failure is left for the page to meet
export function preload<Value>(loader: Loader<Value>, key: string) {
  if (key) fetchKept(loader, key).catch(() => undefined)
}

// Forget every kept answer, after an import has changed the archive: an answer
// about the archive before it must not be shown again.
export function forgetKept() {
  entries = new WeakMap()
  generation += 1
}
