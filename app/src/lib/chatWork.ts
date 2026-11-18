// What the server does while it answers a question, as steps: each status
// line it sends is a step, the one before it then done; a conversation that
// is read part by part counts its parts on the step that reads it. When the
// answer starts, the steps fold into one line, "Worked for 6s", which the
// answer keeps. Nothing here keeps state or talks to the server.

export type WorkStep = {
  text: string
  state: 'active' | 'done'
  // parts read so far and parts in all, while a conversation is read
  read?: number
  of?: number
}

export type Work = { startedAt: number; steps: WorkStep[] }

// what an answer keeps of the work: how long it took and its steps
export type WorkSummary = { seconds: number; steps: string[] }

// one part of a reading, as the stream's progress event gives it
export type PartEvent = { part: number; of: number; state: string }

export const startWork = (now: number): Work => ({ startedAt: now, steps: [] })

const finished = (step: WorkStep): WorkStep => ({ ...step, state: 'done' })

export function withStatus(work: Work, text: string): Work {
  const last = work.steps.at(-1)
  if (last?.text === text) return work
  return { ...work, steps: [...work.steps.map(finished), { text, state: 'active' }] }
}

// A part that is done, read before (cached) or failed has passed: the bar
// moves on either way, and the answer says what it could not read.
export function withProgress(work: Work, event: PartEvent): Work {
  const whole = (value: number) => Number.isInteger(value) && value > 0
  if (!whole(event.part) || !whole(event.of)) return work
  const passed = event.state === 'done' || event.state === 'cached' || event.state === 'failed'
  const steps = work.steps.length ? work.steps : [{ text: 'Reading a conversation', state: 'active' as const }]
  const last = steps[steps.length - 1]
  const read = Math.min(event.of, (last.read ?? 0) + (passed ? 1 : 0))
  return { ...work, steps: [...steps.slice(0, -1), { ...last, read, of: event.of }] }
}

export function workSummary(work: Work, now: number): WorkSummary | undefined {
  if (!work.steps.length) return undefined
  return {
    seconds: Math.max(1, Math.round((now - work.startedAt) / 1000)),
    steps: work.steps.map((step) => step.text),
  }
}

// how long the work took: "6s", "1m 20s", "2m"
export function workTime(seconds: number) {
  if (seconds < 60) return `${seconds}s`
  const rest = seconds % 60
  return rest ? `${Math.floor(seconds / 60)}m ${rest}s` : `${seconds / 60}m`
}
