import type {
  AnalysisPayload,
  ArchivePatterns,
  AskResponse,
  ContactNamesSummary,
  ContactsResponse,
  DayMessages,
  Defaults,
  FirstWordsPayload,
  Job,
  MemoriesPayload,
  PreviewPayload,
  ReportFile,
  SearchResponse,
  SetupMarker,
  SetupStatus,
} from '../types'
import { ApiError, apiErrorFrom } from './apiError'
import { readFrame, takeFrames } from './askEvents'

const apiBase = import.meta.env.VITE_RECALL_API_BASE?.replace(/\/$/, '') ?? ''

type QueryValue = string | number | boolean | null | undefined

function queryString(values: Record<string, QueryValue>) {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(values)) {
    if (value !== undefined && value !== null && value !== '') {
      params.set(key, String(value))
    }
  }
  const encoded = params.toString()
  return encoded ? `?${encoded}` : ''
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  })

  if (!response.ok) throw await refusal(response)

  return response.json() as Promise<T>
}

// The error for an answer that failed: the server's sentence, its fixed word
// and the whole body. An answer that is not JSON leaves the HTTP status.
async function refusal(response: Response): Promise<ApiError> {
  const body: unknown = await response.json().catch(() => null)
  return apiErrorFrom(response.status, response.statusText, body)
}

// An earlier turn of a chat. A question carries the conversation that it was
// asked of ('' for everything); one from an older build carries none.
export type AskTurn = { role: 'user' | 'assistant'; content: string; contact?: string }

export type AskInput = {
  messagesPath: string
  question: string
  contact?: string
  model?: string
  // how hard the model is asked to think: one of its reasoning levels, or
  // nothing for the server's own choice
  reasoningEffort?: string
  history?: AskTurn[]
  // made by the app for each question; /api/ask/cancel names it
  requestId?: string
}

type AskStreamHandlers = {
  onStatus?: (text: string) => void
  onDelta?: (text: string) => void
  // the text that the deltas gave so far was not the answer: clear it
  onReset?: () => void
}

function askBody(input: AskInput) {
  return {
    messagesPath: input.messagesPath,
    question: input.question,
    contact: input.contact,
    model: input.model,
    reasoningEffort: input.reasoningEffort || undefined,
    history: input.history,
    requestId: input.requestId,
  }
}

export const recallApi = {
  defaults: () => request<Defaults>('/api/defaults'),

  contacts: (input: { messagesPath?: string; limit?: number }) =>
    request<ContactsResponse>(
      `/api/contacts${queryString({
        messagesPath: input.messagesPath,
        limit: input.limit ?? 80,
      })}`,
    ),

  preview: (input: {
    messagesPath: string
    contact: string
    model: string
    since?: string
    until?: string
  }) =>
    request<PreviewPayload>(
      `/api/preview${queryString({
        messagesPath: input.messagesPath,
        contact: input.contact,
        model: input.model,
        since: input.since,
        until: input.until,
      })}`,
    ),

  reports: () => request<{ reports: ReportFile[] }>('/api/reports'),

  jobs: () => request<{ jobs: Job[] }>('/api/jobs'),

  // offset is the place of the first match to give, for the matches after the first answer
  search: (input: { messagesPath: string; query: string; contact?: string; limit?: number; offset?: number }) =>
    request<SearchResponse>(
      `/api/search${queryString({
        messagesPath: input.messagesPath,
        query: input.query,
        contact: input.contact,
        limit: input.limit ?? 60,
        offset: input.offset,
      })}`,
    ),

  // one day of one conversation: all of it, or the part around a time or from a place in it
  day: (input: {
    messagesPath: string
    contact: string
    day: string
    around?: string
    offset?: number
    limit?: number
  }) =>
    request<DayMessages>(
      `/api/day${queryString({
        messagesPath: input.messagesPath,
        contact: input.contact,
        day: input.day,
        around: input.around,
        offset: input.offset,
        limit: input.limit,
      })}`,
    ),

  ask: (input: AskInput) =>
    request<AskResponse>('/api/ask', {
      method: 'POST',
      body: JSON.stringify(askBody(input)),
    }),

  // Streams the answer: status lines while the server reads, then text deltas
  // as the model writes, then the final payload. Resolves with the same
  // AskResponse shape as ask(). An error event is thrown as an ApiError with
  // the server's sentence and its fixed word. `signal` ends the request from
  // this side; the read then fails with an AbortError.
  askStream: async (input: AskInput, handlers: AskStreamHandlers = {}, signal?: AbortSignal): Promise<AskResponse> => {
    const response = await fetch(`${apiBase}/api/ask/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(askBody(input)),
      signal,
    })
    if (!response.ok || !response.body) throw await refusal(response)

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let finalPayload: AskResponse | null = null

    const handleFrame = (frame: string) => {
      const event = readFrame(frame)
      if (!event) return
      if (event.type === 'status') {
        if (event.reset) handlers.onReset?.()
        if (event.text) handlers.onStatus?.(event.text)
      } else if (event.type === 'delta') handlers.onDelta?.(event.text)
      else if (event.type === 'done') finalPayload = event.payload
      else throw new ApiError(event.error, { status: 0, code: event.code, body: event.body, retryAfter: event.retryAfter })
    }

    try {
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        const taken = takeFrames(buffer + decoder.decode(value, { stream: true }))
        buffer = taken.rest
        taken.frames.forEach(handleFrame)
      }
      if (buffer.trim()) handleFrame(buffer)
    } finally {
      // an error event ends the reading before the stream ends: let the connection go
      reader.cancel().catch(() => undefined)
    }

    if (!finalPayload) throw new Error('The answer ended before it was whole. Ask again.')
    return finalPayload
  },

  // Stops a question that is being answered. `cancelled` is false when the
  // server no longer holds a question under that id.
  cancelAsk: (requestId: string) =>
    request<{ cancelled: boolean }>('/api/ask/cancel', {
      method: 'POST',
      body: JSON.stringify({ requestId }),
    }),

  analysis: (input: { messagesPath: string; model: string; reportPath: string; contact?: string }) =>
    request<{ analysis: AnalysisPayload; cached?: boolean }>(
      `/api/analysis${queryString({
        messagesPath: input.messagesPath,
        model: input.model,
        reportPath: input.reportPath,
        contact: input.contact,
      })}`,
    ),

  // a report is written from the archive as it is; it names no database
  createAnalyzeJob: (input: {
    messagesPath: string
    outDir: string
    contact: string
    model: string
    since?: string
    until?: string
  }) =>
    request<{ job: Job }>('/api/jobs', {
      method: 'POST',
      body: JSON.stringify({ action: 'analyze', ...input }),
    }),

  memories: (input: { messagesPath?: string }) =>
    request<{ memories: MemoriesPayload }>(
      `/api/memories${queryString({ messagesPath: input.messagesPath })}`,
    ),

  overview: (input: { messagesPath?: string }) =>
    request<{ overview: ArchivePatterns }>(
      `/api/overview${queryString({ messagesPath: input.messagesPath })}`,
    ),

  refreshContactNames: () =>
    request<{ contactNames: ContactNamesSummary; message?: string }>('/api/contact-names', {
      method: 'POST',
      body: JSON.stringify({}),
    }),

  clearPreviewCache: () =>
    request<{ cleared: true }>('/api/cache/preview', {
      method: 'POST',
      body: JSON.stringify({}),
    }),

  job: (id: string) => request<{ job: Job }>(`/api/jobs/${id}`),

  setupStatus: (input: { dbPath?: string; messagesPath?: string; deep?: boolean }) =>
    request<SetupStatus>(
      `/api/setup/status${queryString({
        dbPath: input.dbPath,
        messagesPath: input.messagesPath,
        deep: input.deep ? 1 : undefined,
      })}`,
    ),

  setupComplete: (input: SetupMarker) =>
    request<{ setup: SetupMarker }>('/api/setup/complete', {
      method: 'POST',
      body: JSON.stringify(input),
    }),

  // The one import that can name a database: a copy chosen in the setup
  // guide, for this import only. With none named it reads Messages itself.
  // allowRemoval lets an update take out of the archive more messages than
  // the limit allows; without it the limit holds.
  createSetupImportJob: (input: {
    dbPath?: string
    messagesPath?: string
    includeContacts?: boolean
    allowRemoval?: true
  }) =>
    request<{ job: Job }>('/api/jobs', {
      method: 'POST',
      body: JSON.stringify({ action: 'setup_import', ...input }),
    }),

  firstWords: (input: { messagesPath?: string }) =>
    request<{ firstWords: FirstWordsPayload }>(
      `/api/first-words${queryString({ messagesPath: input.messagesPath })}`,
    ),
}
