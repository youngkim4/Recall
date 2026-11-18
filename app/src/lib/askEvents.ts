// The events of /api/ask/stream. The server sends each as a frame,
// "data: <json>" and an empty line; this reads them, and nothing here talks
// to the server.
import type { AskResponse } from '../types'
import { secondsOf } from './apiError.ts'

export type AskEvent =
  // What the server is doing. With reset, the text that the deltas gave so
  // far was a note of the model and not the answer: it is cleared.
  | { type: 'status'; text: string; reset: boolean }
  | { type: 'delta'; text: string }
  // a part of a conversation that is read inside the question: started, or
  // passed ("done", "cached", "failed"). The name and period it also carries
  // are not taken: the status line before it already says what is read.
  | { type: 'progress'; part: number; of: number; state: string }
  | { type: 'done'; payload: AskResponse }
  // body is the whole event, as the server sent it
  | { type: 'error'; code: string; error: string; retryAfter: number | undefined; body: unknown }

const BETWEEN_FRAMES = '\n\n'
const NO_SENTENCE = 'The question could not be answered.'

// The whole frames in what has arrived so far, and the part of a frame that
// is still on its way.
export function takeFrames(buffer: string): { frames: string[]; rest: string } {
  const pieces = buffer.split(BETWEEN_FRAMES)
  return { frames: pieces.slice(0, -1).filter(Boolean), rest: pieces[pieces.length - 1] }
}

function dataOf(frame: string): Record<string, unknown> | null {
  const line = frame.split('\n').find((one) => one.startsWith('data:'))
  if (!line) return null
  try {
    const data: unknown = JSON.parse(line.slice(5))
    return data && typeof data === 'object' && !Array.isArray(data) ? (data as Record<string, unknown>) : null
  } catch {
    // a frame that is not JSON carries nothing this build can read
    return null
  }
}

const text = (value: unknown) => (typeof value === 'string' ? value : '')

// The event that a frame carries, or null for a frame with nothing to act
// on: a comment, a kind this build does not know, an event with no content.
export function readFrame(frame: string): AskEvent | null {
  const data = dataOf(frame)
  if (!data) return null
  if (data.type === 'status') {
    const event = { type: 'status' as const, text: text(data.text), reset: data.reset === true }
    return event.text || event.reset ? event : null
  }
  if (data.type === 'delta') return text(data.text) ? { type: 'delta', text: text(data.text) } : null
  if (data.type === 'progress') {
    const { part, of } = data
    if (typeof part !== 'number' || typeof of !== 'number') return null
    return { type: 'progress', part, of, state: text(data.state) }
  }
  if (data.type === 'done') {
    return data.payload && typeof data.payload === 'object' ? { type: 'done', payload: data.payload as AskResponse } : null
  }
  if (data.type === 'error') {
    return {
      type: 'error',
      code: text(data.code),
      error: text(data.error) || NO_SENTENCE,
      retryAfter: secondsOf(data.retry_after),
      body: data,
    }
  }
  return null
}
