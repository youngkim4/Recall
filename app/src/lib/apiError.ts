// What the server says when it refuses: one plain sentence, and often a
// fixed word for why. The sentence is the message of the error.

type Fields = {
  // the HTTP status; 0 for an error that came inside a stream
  status: number
  code?: string
  // the whole answer, as the server sent it
  body?: unknown
  // seconds to wait before asking again, where the server says
  retryAfter?: number
}

export class ApiError extends Error {
  status: number
  code?: string
  body?: unknown
  retryAfter?: number

  constructor(message: string, fields: Fields) {
    super(message)
    this.name = 'ApiError'
    this.status = fields.status
    this.code = fields.code
    this.body = fields.body
    this.retryAfter = fields.retryAfter
  }
}

const textOf = (value: unknown) => (typeof value === 'string' && value ? value : undefined)

// a count of seconds, or nothing when the server gave no such number
export const secondsOf = (value: unknown) =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined

// The error for an answer that failed. An answer with no sentence in it
// leaves the HTTP status to say what happened.
export function apiErrorFrom(status: number, statusText: string, body: unknown): ApiError {
  const said = body && typeof body === 'object' && !Array.isArray(body) ? (body as Record<string, unknown>) : {}
  return new ApiError(textOf(said.error) ?? `${status} ${statusText}`.trim(), {
    status,
    code: textOf(said.code),
    body,
    retryAfter: secondsOf(said.retry_after),
  })
}

// the fixed word of an error, or '' when it has none
export function errorCode(error: unknown) {
  return error instanceof ApiError ? (error.code ?? '') : ''
}

// The sentence to show for something that went wrong. What the server said
// is shown as it is. A request that reached no server has only the words of
// the browser, which are not for the reader: `fallback` is shown then.
export function errorSentence(error: unknown, fallback: string) {
  if (!(error instanceof Error) || error instanceof TypeError) return fallback
  return error.message || fallback
}
