import { useCallback, useEffect, useState } from 'react'

export type Load<Value> = { state: 'loading' } | { state: 'failed' } | { state: 'ready'; value: Value }

// what was loaded, and the key and attempt it answers
type Held<Value> = { key: string; attempt: number; load: Load<Value> }

const LOADING = { state: 'loading' } as const
const FAILED = { state: 'failed' } as const

// Loads one thing for a key, such as the path of an archive. An answer
// belongs to its key: under another key the load starts again, so nothing of
// one archive is shown as another's, and an old failure is not shown for a
// new load. retry starts over after a failure. An empty key has nothing to
// load and counts as failed. The loader must be the same function on every
// render.
export function useLoad<Value>(key: string, loader: (key: string) => Promise<Value>) {
  const [held, setHeld] = useState<Held<Value> | null>(null)
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    if (!key) return
    let cancelled = false
    const settle = (load: Load<Value>) => {
      if (!cancelled) setHeld({ key, attempt, load })
    }
    // a loader that throws before it returns a promise fails like one that rejects
    new Promise<Value>((resolve) => resolve(loader(key))).then(
      (value) => settle({ state: 'ready', value }),
      () => settle(FAILED),
    )
    return () => {
      cancelled = true
    }
  }, [key, loader, attempt])

  const retry = useCallback(() => setAttempt((current) => current + 1), [])

  const answered = held !== null && held.key === key && held.attempt === attempt
  const load: Load<Value> = !key ? FAILED : answered ? held.load : LOADING
  return { load, retry }
}
