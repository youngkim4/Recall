// The figures of the last conversation that the Analyze view holds in
// memory, and the rule for when they may be shown.
import type { PreviewPayload } from '../types'

export type Held = {
  // the request the figures answer
  key: string
  // the archive and conversation they belong to
  subject: string
  // how many times the saved figures had been cleared when these arrived
  generation: number
  payload: PreviewPayload | null
}

export const NOTHING_HELD: Held = { key: '', subject: '', generation: 0, payload: null }

// What is shown for a request. While new dates or another quality are worked
// out, the figures of the same conversation stay up. Figures from before the
// saved ones were cleared are never shown: an import wrote another archive,
// and they are of the one before it.
export function shownPreview(held: Held, asked: Omit<Held, 'payload'>): { payload: PreviewPayload | null; settled: boolean } {
  if (held.generation !== asked.generation) return { payload: null, settled: false }
  const fresh = Boolean(asked.key) && held.key === asked.key
  const sameSubject = Boolean(asked.subject) && held.subject === asked.subject
  return {
    payload: fresh || sameSubject ? held.payload : null,
    settled: fresh && held.payload !== null,
  }
}
