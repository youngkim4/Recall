import { useEffect, useState } from 'react'
import type { Load } from './useLoad'

// A value that follows another once it has held still for `wait` ms: words
// that are counted once typing pauses, not on each key.
export function useSettled<Value>(value: Value, wait: number): Value {
  const [settled, setSettled] = useState(value)
  useEffect(() => {
    const timer = window.setTimeout(() => setSettled(value), wait)
    return () => window.clearTimeout(timer)
  }, [value, wait])
  return settled
}

// The last answer a load gave, kept on the page while the next one loads: a
// chart that follows a filter changes in place and does not blink to its
// placeholder on each change. Null until the first answer.
export function useLastReady<Value>(load: Load<Value>): Value | null {
  const [last, setLast] = useState<Value | null>(null)
  if (load.state === 'ready' && load.value !== last) setLast(load.value)
  return load.state === 'ready' ? load.value : last
}
