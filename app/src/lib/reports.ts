// Saved reports: one to a conversation, written as Markdown with companion files.

import type { Contact, ReportFile } from '../types'
import { reportTitle } from './conversations.ts'

export type ReportEntry = {
  // the conversation the report is about
  key: string
  // the file that is read: the Markdown one when there is one
  report: ReportFile
  title: string
  // when the newest of its files was written
  writtenAt?: string
  files: ReportFile[]
}

const newestFirst = (a?: string, b?: string) => String(b || '').localeCompare(String(a || ''))

// Every report once, newest first, however many files it was saved as.
export function reportEntries(reports: ReportFile[]): ReportEntry[] {
  const byConversation = reports.reduce((groups, report) => {
    const key = report.contact || report.path.replace(/\.(html|md)$/i, '')
    return new Map(groups).set(key, [...(groups.get(key) ?? []), report])
  }, new Map<string, ReportFile[]>())

  return [...byConversation]
    .map(([key, files]) => {
      const sorted = [...files].sort((a, b) => newestFirst(a.updatedAt, b.updatedAt))
      const report = sorted.find((file) => file.kind === 'md') ?? sorted[0]
      return { key, report, title: reportTitle(report), writtenAt: sorted[0].updatedAt, files: sorted }
    })
    .sort((a, b) => newestFirst(a.writtenAt, b.writtenAt))
}

// A report goes by the name of its conversation. The server resolves saved
// names; the name made for an unnamed group comes from the conversation list.
export function nameReports(reports: ReportFile[], contacts: Contact[]): ReportFile[] {
  const names = new Map(contacts.map((contact) => [contact.chat_id, contact.display_name || contact.displayName || '']))
  return reports.map((report) =>
    report.displayName ? report : { ...report, displayName: names.get(report.contact) || '' },
  )
}

// "turning_point" -> "Turning point"
export function eventKind(category?: string) {
  const words = (category || '').replace(/_/g, ' ').trim()
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : ''
}

// each mark that can open a quote, with the marks that can close it
const CLOSERS: Record<string, string> = { '"': '"”', '“': '"”', "'": "'’", '‘': "'’" }
// a word that begins with an apostrophe of its own: 'tis, 'cause, '90s
const ELIDED = /^['‘](?:tis|twas|twere|twill|til|till|cause|cuz|cos|coz|em|bout|round|nuff|sup|kay|n|\d0s)\b/i
const WORD = /[\p{L}\p{N}]/u
// what a mark follows when it opens a quote inside a sentence
const LEAD = /[\s([{—–-]/

// What a mark inside a text does to the depth of quoting: 1 opens a quote,
// -1 closes one, 0 is neither, like the apostrophe in "it's".
function turn(marks: string[], at: number, single: boolean) {
  const mark = marks[at]
  const before = marks[at - 1]
  const after = marks[at + 1]
  const inWord = WORD.test(before) && WORD.test(after)
  if (!(single ? "'‘’" : '"“”').includes(mark) || (single && inWord)) return 0
  if (mark === '“' || mark === '‘') return 1
  if (mark === '”' || mark === '’') return -1
  // a straight mark opens after a space and closes after a word
  if (LEAD.test(before)) return WORD.test(after) ? 1 : 0
  return WORD.test(after) ? 0 : -1
}

// Whether the first and last marks of a text are one pair around all of it.
// In "Hello" she said "bye" they are not: the first closes after one word.
function wrapped(text: string) {
  const marks = [...text]
  const closers = CLOSERS[marks[0]]
  if (marks.length < 2 || !closers?.includes(marks[marks.length - 1]) || ELIDED.test(text)) return false
  const single = closers.includes("'")
  const end = marks.slice(1, -1).reduce(
    (state, _mark, index) => {
      if (state.closed) return state
      const depth = state.depth + turn(marks, index + 1, single)
      return { depth, closed: depth === 0 }
    },
    { depth: 1, closed: false },
  )
  return !end.closed
}

// A quoted message as plain words: the pair of quote marks it came wrapped in is dropped.
export function plainQuote(quote?: string) {
  const text = (quote || '').trim()
  return wrapped(text) ? [...text].slice(1, -1).join('').trim() : text
}
