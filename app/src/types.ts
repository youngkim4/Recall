import type { ActivityPoint, ActivityUnit } from './lib/activity'
import type { Daily } from './lib/days'

// Settings is no view: it is a sheet over the view in use
export type ViewKey = 'home' | 'analyze' | 'insights' | 'explore' | 'ask' | 'reports' | 'jobs'

// How carefully a model reads, as the server ranks it.
export type ModelLevel = 'low' | 'medium' | 'high' | 'top'

// A model that the server offers: its id, its maker's name for it, its level.
export type ModelChoice = {
  id: string
  label: string
  level: ModelLevel
  // how hard the model can be asked to think about a chat question, least
  // first; none from a server that does not say
  reasoningLevels?: string[]
  // the level of a question that asks for none
  reasoningDefault?: string | null
}

export type Defaults = {
  defaultModel: string
  models: string[]
  // the same models with their names and levels, lowest level first
  modelChoices?: ModelChoice[]
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
  // the parts of the plan, and how many of them an earlier run already read;
  // the cost of those is not in estimated_cost
  periods?: number
  periods_cached?: number
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
  // From a search: the ranges of `text` the words matched, [start, end) in
  // UTF-16 units. Empty for a search with no words; missing from an older
  // server and from anything that is not a search.
  ranges?: Array<[number, number]>
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
  // the place of the first result among all the matches
  offset?: number
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
    // what was said on the first day, as a moment of this day quotes its
    // day; absent from a server that does not quote it
    preview?: string
    previewFromMe?: boolean | null
    previewSender?: string
    previewAt?: string | null
    previewId?: string
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

// An answer of an older build, kept in a saved chat, has only the first six keys.
export type AskResponse = {
  question: string
  contact?: string
  terms: string[]
  answer: string
  mode?: 'ai' | 'local'
  citations: SearchResult[]
  // quotes in the answer that the archive does not hold
  unverifiedQuotes?: string[]
  // the reader stopped the question before the answer ended
  cancelled?: boolean
  // the answer was cut short
  truncated?: boolean
  steps?: unknown[]
  usage?: Record<string, unknown>
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
  // null when the plan of a report cannot be made; estimateError then says why
  estimate: Estimate | null
  estimateError?: string
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
  // The place of the moment in the archive. A report of an older build has
  // none of the four, and its moments open nothing.
  chatId?: string
  // "YYYY-MM-DD"
  day?: string
  // when the quoted message was sent, as `timestamp` of a day message; null with no quote
  at?: string | null
  // the id of the quoted message; '' when there is none
  pickedId?: string
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
  // the story is too long and dense for chapters, so it is shown as plain paragraphs
  shownPlain?: boolean
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
  // `done` of `total`: how far the step is, where the step can tell
  progress?: { step: number; of: number; label: string; done?: number; total?: number }
  // what the job paid the model provider so far, in US dollars
  spent?: { usd: number } | null
  result?: {
    reportPath?: string
    eventsPath?: string | null
    htmlPath?: string | null
    jsonPath?: string
    reportId?: string
    status?: string
    // what went wrong on the way to a report that was still written
    warnings?: string[]
    shownPlain?: boolean
    usage?: Record<string, unknown>
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
  // a fixed word for why a job failed, beside the sentence in error
  code?: string
}

// Where the API key is (GET /api/key). The key itself is never sent back.
export type KeyState = {
  present: boolean
  // 'environment': set where the server starts, and a saved key would not be
  // used; 'file': saved by the app; null: there is none
  source: 'environment' | 'file' | null
  // where it is, in words
  place?: string
}

// What getting chat ready builds, in this order (GET /api/first-run)
export type FirstRunStepName = 'meaning' | 'memory' | 'people'

export type FirstRunStep = {
  name: FirstRunStepName
  // 'behind' and 'partial' are begun and not ready
  state: 'none' | 'behind' | 'partial' | 'ready'
  // an upper bound in US dollars; 0 when `error` says why it is not known
  estimatedCost: number
  error: string | null
  code: string | null
}

export type FirstRunStatus = {
  key: Pick<KeyState, 'present' | 'source'>
  archive: { holdsMessages: boolean; conversations: number; messages: number }
  steps: FirstRunStep[]
  // the sum over the steps that are not ready: an upper bound
  estimatedCost: number
  ready: boolean
  // the first_run job that runs now, or null
  jobId: string | null
}
