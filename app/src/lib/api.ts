import type {
  AnalysisPayload,
  AskResponse,
  ContactNamesSummary,
  ContactsResponse,
  Defaults,
  Job,
  PreviewPayload,
  ReportFile,
  SearchResponse,
} from '../types'

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

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`
    try {
      const body = (await response.json()) as { error?: string }
      message = body.error || message
    } catch {
      // Keep the HTTP status as the fallback error.
    }
    throw new Error(message)
  }

  return response.json() as Promise<T>
}

type AskInput = {
  messagesPath: string
  question: string
  contact?: string
  model?: string
  limit?: number
  history?: Array<{ role: 'user' | 'assistant'; content: string }>
}

type AskStreamHandlers = {
  onStatus?: (text: string) => void
  onDelta?: (text: string) => void
}

function askBody(input: AskInput) {
  return {
    messagesPath: input.messagesPath,
    question: input.question,
    contact: input.contact,
    model: input.model,
    limit: input.limit ?? 8,
    history: input.history,
  }
}

export const recallApi = {
  defaults: () => request<Defaults>('/api/defaults'),

  contacts: (input: { messagesPath?: string; dbPath?: string; limit?: number }) =>
    request<ContactsResponse>(
      `/api/contacts${queryString({
        messagesPath: input.messagesPath,
        dbPath: input.dbPath,
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

  search: (input: { messagesPath: string; query: string; contact?: string; limit?: number }) =>
    request<SearchResponse>(
      `/api/search${queryString({
        messagesPath: input.messagesPath,
        query: input.query,
        contact: input.contact,
        limit: input.limit ?? 60,
      })}`,
    ),

  ask: (input: AskInput) =>
    request<AskResponse>('/api/ask', {
      method: 'POST',
      body: JSON.stringify(askBody(input)),
    }),

  // Streams the answer: status lines while retrieving, then text deltas as the
  // model writes, then the final payload. Resolves with the same AskResponse
  // shape as ask().
  askStream: async (input: AskInput, handlers: AskStreamHandlers): Promise<AskResponse> => {
    const response = await fetch(`${apiBase}/api/ask/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(askBody(input)),
    })
    if (!response.ok || !response.body) {
      throw new Error(`${response.status} ${response.statusText}`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let finalPayload: AskResponse | null = null

    const handleFrame = (frame: string) => {
      const line = frame.split('\n').find((l) => l.startsWith('data: '))
      if (!line) return
      const event = JSON.parse(line.slice(6)) as {
        type: 'status' | 'delta' | 'done' | 'error'
        text?: string
        payload?: AskResponse
        error?: string
      }
      if (event.type === 'status' && event.text) handlers.onStatus?.(event.text)
      else if (event.type === 'delta' && event.text) handlers.onDelta?.(event.text)
      else if (event.type === 'done' && event.payload) finalPayload = event.payload
      else if (event.type === 'error') throw new Error(event.error || 'Ask failed.')
    }

    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let boundary = buffer.indexOf('\n\n')
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary)
        buffer = buffer.slice(boundary + 2)
        handleFrame(frame)
        boundary = buffer.indexOf('\n\n')
      }
    }
    if (buffer.trim()) handleFrame(buffer)

    if (!finalPayload) throw new Error('The answer stream ended unexpectedly.')
    return finalPayload
  },

  analysis: (input: { messagesPath: string; model: string; reportPath: string; contact?: string }) =>
    request<{ analysis: AnalysisPayload; cached?: boolean }>(
      `/api/analysis${queryString({
        messagesPath: input.messagesPath,
        model: input.model,
        reportPath: input.reportPath,
        contact: input.contact,
      })}`,
    ),

  createAnalyzeJob: (input: {
    dbPath: string
    messagesPath: string
    outDir: string
    contact: string
    model: string
    html: boolean
    extractFirst: boolean
    since?: string
    until?: string
  }) =>
    request<{ job: Job }>('/api/jobs', {
      method: 'POST',
      body: JSON.stringify({ action: 'analyze', ...input }),
    }),

  createExportJob: (input: { dbPath: string; messagesPath: string }) =>
    request<{ job: Job }>('/api/jobs', {
      method: 'POST',
      body: JSON.stringify({ action: 'export', ...input }),
    }),

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
}
