import assert from 'node:assert/strict'
import { test } from 'node:test'
import { eventKind, nameReports, plainQuote, reportEntries } from './reports.ts'

const file = (name: string, kind: string, contact: string, updatedAt: string, displayName = '') => ({
  name,
  path: `/reports/${name}`,
  kind,
  contact,
  updatedAt,
  displayName,
})

test('a report is its Markdown file, with its other files as companions', () => {
  const entries = reportEntries([
    file('analysis_a.html', 'html', '+15550000001', '2026-11-12T10:00:05'),
    file('analysis_a.md', 'md', '+15550000001', '2026-11-12T10:00:00', 'Avery Stone'),
    file('analysis_b.md', 'md', 'chat77', '2026-11-14T09:00:00'),
  ])

  // newest first
  assert.deepEqual(entries.map((entry) => entry.key), ['chat77', '+15550000001'])
  assert.equal(entries[1].report.kind, 'md')
  assert.equal(entries[1].title, 'Avery Stone')
  // the time it was written is the newest of its files
  assert.equal(entries[1].writtenAt, '2026-11-12T10:00:05')
})

test('a report with no Markdown file is still listed', () => {
  const entries = reportEntries([file('analysis_c.html', 'html', '+15550000003', '2026-11-12T10:00:00')])

  assert.equal(entries.length, 1)
  assert.equal(entries[0].report.kind, 'html')
})

test('a report takes the name its conversation goes by', () => {
  const reports = [file('analysis_b.md', 'md', 'chat77', '2026-11-14T09:00:00'), file('analysis_a.md', 'md', '+15550000001', '2026-11-12T10:00:00', 'Avery Stone')]
  const contacts = [
    { chat_id: 'chat77', display_name: 'Avery Stone and Blake Reyes' },
    { chat_id: '+15550000001', display_name: 'Avery' },
  ]

  const named = nameReports(reports, contacts)

  // a group nobody named gets the name the conversation list made for it
  assert.equal(named[0].displayName, 'Avery Stone and Blake Reyes')
  // a name the server already resolved is kept
  assert.equal(named[1].displayName, 'Avery Stone')
})

test('the kind of a moment is written as words', () => {
  assert.equal(eventKind('turning_point'), 'Turning point')
  assert.equal(eventKind('milestone'), 'Milestone')
  assert.equal(eventKind(''), '')
  assert.equal(eventKind(undefined), '')
})

test('a quoted message loses the quote marks it came wrapped in', () => {
  assert.equal(plainQuote('"see you at 8"'), 'see you at 8')
  assert.equal(plainQuote('“see you at 8”'), 'see you at 8')
  assert.equal(plainQuote("'see you at 8'"), 'see you at 8')
  assert.equal(plainQuote('‘see you at 8’'), 'see you at 8')
  assert.equal(plainQuote('  it\'s "fine"  '), 'it\'s "fine"')
  assert.equal(plainQuote(undefined), '')
})

test('marks inside a wrapped message stay where they are', () => {
  assert.equal(plainQuote('"He said "hi" to me"'), 'He said "hi" to me')
  assert.equal(plainQuote("'it's fine, don't worry'"), "it's fine, don't worry")
  assert.equal(plainQuote('“it’s “fine” now”'), 'it’s “fine” now')
})

test('a message that only starts and ends with quoted words keeps every mark', () => {
  assert.equal(plainQuote('"Hello" she said "bye"'), '"Hello" she said "bye"')
  assert.equal(plainQuote('"Hello," she said, "bye."'), '"Hello," she said, "bye."')
  assert.equal(plainQuote('“one” and “two”'), '“one” and “two”')
  assert.equal(plainQuote("'yes' or 'no'"), "'yes' or 'no'")
})

test('an apostrophe that begins a word is not a quote mark', () => {
  assert.equal(plainQuote("'Tis the season'"), "'Tis the season'")
  assert.equal(plainQuote("'cause I said so'"), "'cause I said so'")
  assert.equal(plainQuote("'90s music, nothin'"), "'90s music, nothin'")
})

test('two marks of different kinds are not a pair', () => {
  assert.equal(plainQuote('"see you at 8\''), '"see you at 8\'')
  assert.equal(plainQuote('"'), '"')
  assert.equal(plainQuote('""'), '')
})
