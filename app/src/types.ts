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
  setupCompleted?: boolean
}

export type SetupDbStatus =
  | 'readable'
  | 'fda_blocked'
  | 'missing'
  | 'empty'
  | 'invalid'
  | 'locked'
  | 'error'

export type SetupState =
  | 'ready'
  | 'needs_export'
  | 'needs_permission'
  | 'no_messages'
  | 'db_locked'
  | 'db_invalid'

export type SetupMarker = {
  completed?: boolean
  skipped?: boolean
  firstWordsShown?: boolean
  pickedDbPath?: string
}

export type SetupStatus = {
  state: SetupState
  db: {
    status: SetupDbStatus
    kind: 'live' | 'copy'
    path: string
    detail?: string
    approxMessages?: number | null
    conversations?: number | null
    firstYear?: number | null
    lastYear?: number | null
  }
  export: { exists: boolean; path: string }
  setup: SetupMarker
}

export type SetupImportResult = {
  messages: number
  conversations: number
  firstYear?: number | null
  lastYear?: number | null
  contactsSkipped?: boolean
  recovered?: number
}

export type FirstWordsEntry = {
  person: string
  chatId: string
  timestamp: string
  yearsAgo: number
  direction: string
  text: string
  reply?: { text: string; direction: string; timestamp: string } | null
  messageCount: number
}

export type FirstWordsPayload = {
  entries: FirstWordsEntry[]
  signature: string
  totals: {
    messages: number
    people: number
    firstYear?: number | null
    lastYear?: number | null
  }
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
  senderName?: string
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

export type MemoriesPayload = {
  onThisDay: Array<{
    chatId: string
    name: string
    year: number
    yearsAgo: number
    count: number
    preview: string
  }>
  anniversaries: Array<{
    chatId: string
    name: string
    years: number
    date: string
    inDays: number
    count: number
  }>
  reconnect: Array<{
    chatId: string
    name: string
    count: number
    quietDays: number
    lastDate: string
  }>
}

export type SemanticStatus = {
  state: 'none' | 'stale' | 'fresh'
  windows?: number
  builtAt?: string
  model?: string
  estimate?: {
    windows: number
    tokens: number
    estimatedCost: number
  }
}

export type AskResponse = {
  question: string
  contact?: string
  terms: string[]
  answer: string
  mode?: 'ai' | 'local'
  citations: SearchResult[]
}

export type Dynamics = {
  balanceLifetime?: number | null
  balanceRecent?: number | null
  initiationLifetime?: number | null
  initiationRecent?: number | null
  volumeTrendPct?: number
  quietDays?: number
  topSpeakers?: Array<{ name: string; count: number; share: number }>
}

export type PreviewPayload = {
  stats: PreviewStats
  estimate: Estimate
  monthly: MonthlyPoint[]
  recentMessages: RecentMessage[]
  dynamics?: Dynamics
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
  progress?: { step: number; of: number; label: string }
  result?: {
    reportPath?: string
    eventsPath?: string
    htmlPath?: string
    analysis?: AnalysisPayload
    reports?: ReportFile[]
    messages?: number
    conversations?: number
    firstYear?: number | null
    lastYear?: number | null
    contactsSkipped?: boolean
    recovered?: number
  } | null
  error?: string | null
}
