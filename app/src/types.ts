export type ViewKey = 'home' | 'analyze' | 'explore' | 'ask' | 'reports' | 'jobs' | 'settings'

export type Defaults = {
  defaultModel: string
  models: string[]
  dbPath: string
  messagesPath: string
  outDir: string
  hasDb: boolean
  hasMessages: boolean
  reports: ReportFile[]
  contactNames?: ContactNamesSummary
}

export type RuntimePaths = {
  dbPath: string
  messagesPath: string
  outDir: string
}

export type ContactNamesSummary = {
  count?: number
  updatedAt?: string
  exportedAt?: string
  message?: string
  exists?: boolean
  invalid?: boolean
}

export type PreviewCacheSummary = {
  count: number
  lastSavedAt: string
}

export type Contact = {
  chat_id: string
  display_name?: string
  displayName?: string
  is_group?: boolean | number
  first_msg?: string
  last_msg?: string
  message_count?: number
}

export type ContactsResponse = {
  contacts: Contact[]
  source: 'messages' | 'database'
  messagesPath?: string
  dbPath?: string
  contactNameCount?: number
}

export type ReportFile = {
  name: string
  path: string
  kind: 'md' | 'html' | string
  contact: string
  displayName?: string
  updatedAt?: string
  size?: number
}

export type PreviewStats = {
  chatId?: string
  totalMessages?: number
  sentCount?: number
  receivedCount?: number
  unknownDirectionCount?: number
  firstTimestamp?: string
  lastTimestamp?: string
  activeDays?: number
  avgMessagesPerDay?: number
  busiestDay?: string
  busiestDayCount?: number
  longestGapDays?: number
  attachments?: Record<string, number>
  reactions?: Record<string, number>
}

export type Estimate = {
  msg_count?: number
  input_tokens?: number
  output_tokens?: number
  estimated_cost?: number
  needs_chunking?: boolean
  token_budget?: number
  long_context_pricing?: boolean
  years?: number[]
}

export type MonthlyPoint = {
  month: string
  total: number
  sent: number
  received: number
  sentRatio: number
}

export type RecentMessage = {
  timestamp: string
  text: string
  isFromMe?: boolean | number | string
}

export type SearchResult = {
  messageId?: string
  chatId: string
  displayName?: string
  timestamp: string
  sender?: string
  text: string
  isFromMe?: boolean | number | string
}

export type SearchResponse = {
  query: string
  contact?: string
  count: number
  results: SearchResult[]
}

export type AskResponse = {
  question: string
  contact?: string
  terms: string[]
  answer: string
  mode?: 'ai' | 'local'
  citations: SearchResult[]
}

export type PreviewPayload = {
  stats: PreviewStats
  estimate: Estimate
  monthly: MonthlyPoint[]
  recentMessages: RecentMessage[]
  cached?: boolean
}

export type AnalysisEvent = {
  date: string
  title: string
  detail: string
  category?: string
  score?: number | null
  quote?: string
}

export type AnalysisPayload = PreviewPayload & {
  contact: string
  contactDisplayName?: string
  generatedAt?: string
  summary?: string
  summaryHtml?: string
  events?: AnalysisEvent[]
  patterns?: Array<{ label: string; value: string; detail: string }>
  media?: {
    attachments?: Record<string, number>
    reactions?: Record<string, number>
  }
  files?: Array<{ label: string; path: string; name: string; kind: string; size?: number }>
}

export type Job = {
  id: string
  action: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  createdAt: string
  updatedAt: string
  logs: Array<string | { time?: string; message?: string }>
  result?: {
    reportPath?: string
    eventsPath?: string
    htmlPath?: string
    analysis?: AnalysisPayload
    reports?: ReportFile[]
  } | null
  error?: string | null
}
