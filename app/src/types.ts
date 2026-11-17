import type { ActivityPoint, ActivityUnit } from './lib/activity'
import type { Daily } from './lib/days'

export type ViewKey = 'home' | 'analyze' | 'explore' | 'ask' | 'reports' | 'jobs' | 'settings'

export type Defaults = {
  defaultModel: string
  models: string[]
  messagesPath: string
  outDir: string
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
  // holdsMessages: the archive file has a message in it, so the guide is an update
  export: { exists: boolean; path: string; holdsMessages?: boolean }
  setup: SetupMarker
}

export type SetupImportResult = {
  messages: number
  conversations: number
  firstYear?: number | null
  lastYear?: number | null
  contactsSkipped?: boolean
  // the archive is written, and a later step of the import failed
  warning?: string
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

// where the archive is kept and where reports are saved; no database is among them
export type RuntimePaths = {
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
  // how many of those you sent, and how many the other side did; a message
  // that does not say who sent it is in neither
  sent_count?: number
  received_count?: number
}

// Size and span of the whole archive. The contacts list is capped, so totals
// come from here and never from adding up that list.
export type ArchiveSummary = {
  messages: number
  conversations: number
  firstMessage: string | null
  lastMessage: string | null
}

// The shape of the whole archive, for the Overview.
export type ArchivePatterns = {
  // every month in both voices, silent months included
  monthly: ActivityPoint[]
  // how many messages each day holds
  daily: Daily
  // how many fall in each hour of each weekday: seven rows from Monday, twenty-four hours from midnight
  rhythm: number[][]
}

export type ContactsResponse = {
  contacts: Contact[]
  source: 'messages' | 'database'
  messagesPath?: string
  contactNameCount?: number
  archive?: ArchiveSummary
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
  // cleaned for display; empty when the message was only an attachment
  text: string
  // "Photo", "3 photos", "2 attachments", or '' when it carried none
  attachment?: string
  // null when the export does not say who sent it
  isFromMe?: boolean | number | string | null
  // in a group: the handle and saved name of whoever wrote it
  sender?: string
  senderName?: string
}

export type SearchResult = {
  messageId?: string
  chatId: string
  displayName?: string
  senderName?: string
  // null for a message the export gives no readable time for
  timestamp: string | null
  sender?: string
  text: string
  attachment?: string
  isFromMe?: boolean | number | string
}

// One day of one conversation. A very long day is given in part.
export type DayMessages = {
  contact: string
  // "YYYY-MM-DD"
  day: string
  // how many messages the day holds, and where among them `messages` starts
  total: number
  offset: number
  // the nearest days that hold messages; null at either end of the conversation
  previousDay: string | null
  nextDay: string | null
  messages: SearchResult[]
}

export type SearchResponse = {
  query: string
  contact?: string
  // how many are in results, and how many matched in all
  count: number
  total: number
  results: SearchResult[]
}

// name is the saved name or '': conversationName() labels the rest.
export type MemoriesPayload = {
  onThisDay: Array<{
    chatId: string
    name: string
    year: number
    // the day itself, "YYYY-MM-DD"
    day: string
    yearsAgo: number
    count: number
    preview: string
    // null when the export does not say who sent it
    previewFromMe: boolean | null
    // in a group: the saved name of whoever wrote it, or ''
    previewSender: string
    // when the quoted message was sent; null when there is no quote
    previewAt: string | null
    // the quoted message's id, as a message of /api/day carries it; '' when
    // there is no quote, and absent from a server that does not say
    previewId?: string
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
    // days without a message, counted to the end of the archive, not to today
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
  // groups only: how many people have written, and each of them, busiest
  // first; a very large group is cut short. name is the saved name or '';
  // share is of the messages you received.
  speakerCount?: number
  speakers?: Array<{ handle: string; name: string; count: number; share: number }>
}

export type PreviewPayload = {
  stats: PreviewStats
  estimate: Estimate
  // the months the chosen dates cover, silent ones included
  monthly: MonthlyPoint[]
  // the same stretch for the chart: by day when it is short, by month when long
  activity: { unit: ActivityUnit; points: ActivityPoint[] }
  // how many messages each day of the conversation holds, whatever dates were chosen
  daily: Daily
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
  // the conversation a report is written about; '' for work on the whole archive
  contact?: string
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
    warning?: string
  } | null
  error?: string | null
}
