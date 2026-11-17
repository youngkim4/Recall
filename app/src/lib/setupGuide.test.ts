import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { Job, SetupDbStatus, SetupState, SetupStatus } from '../types.ts'
import {
  canImport,
  failedHeading,
  foundTheSame,
  frameStep,
  guideStep,
  importRequest,
  importView,
  interruptedReason,
  noNotes,
  noted,
  notStartedJob,
  revealsFirstWords,
  stepForDatabase,
  whyNotThisCopy,
  type GuideStep,
} from './setupGuide.ts'

const EVERY_STATUS: Record<SetupDbStatus, GuideStep> = {
  readable: 'import',
  locked: 'import',
  fda_blocked: 'permission',
  missing: 'no_messages',
  empty: 'no_messages',
  invalid: 'db_problem',
  error: 'db_problem',
}
const STATUSES = Object.keys(EVERY_STATUS) as SetupDbStatus[]

// what the server says of setup when there is no archive yet
const FIRST_SETUP: Record<SetupDbStatus, SetupState> = {
  readable: 'needs_export',
  fda_blocked: 'needs_permission',
  missing: 'no_messages',
  empty: 'no_messages',
  locked: 'db_locked',
  invalid: 'db_invalid',
  error: 'db_invalid',
}

// The check of Messages itself, as the server gives it. With an archive there
// the server says "ready", whatever it found in the database.
function checked(db: SetupDbStatus, archive: boolean): SetupStatus {
  return {
    state: archive ? 'ready' : FIRST_SETUP[db],
    db: { status: db, kind: 'live', path: '/Users/reader/Library/Messages/chat.db' },
    export: { exists: archive, path: '/data/messages.csv' },
    setup: {},
  }
}

function copyAt(path: string, db: SetupDbStatus = 'readable'): SetupStatus {
  return { ...checked(db, true), db: { status: db, kind: 'copy', path } }
}

const job = (fields: Partial<Job>): Job => ({
  id: 'job-1',
  action: 'setup_import',
  status: 'running',
  createdAt: '2026-05-01T10:00:00',
  updatedAt: '2026-05-01T10:00:00',
  logs: [],
  ...fields,
})

const shown = (db: SetupDbStatus, archive: boolean) =>
  guideStep({ stage: 'source', job: null, picked: null, live: checked(db, archive) })

test('each thing the check of the database can find has its step', () => {
  for (const status of STATUSES) assert.equal(stepForDatabase(status), EVERY_STATUS[status], status)
})

test('the step fits the last check, for a first setup and for an update', () => {
  for (const status of STATUSES) {
    assert.equal(shown(status, false), EVERY_STATUS[status], `first setup, ${status}`)
    assert.equal(shown(status, true), EVERY_STATUS[status], `update, ${status}`)
  }
})

test('the import is never offered for a database that cannot be read', () => {
  // an archive file is there, so the server says "ready": that does not choose the step
  for (const status of ['fda_blocked', 'missing', 'empty', 'invalid', 'error'] as const) {
    assert.equal(canImport(status), false, status)
    assert.notEqual(shown(status, true), 'import', status)
    assert.notEqual(shown(status, false), 'import', status)
  }
})

test('a busy database can start an import: the server tries it, and tries once more', () => {
  assert.equal(canImport('locked'), true)
  assert.equal(shown('locked', true), 'import')
  assert.equal(shown('locked', false), 'import')
})

test('a first setup opens on its welcome, whatever the database is', () => {
  for (const status of STATUSES) {
    assert.equal(guideStep({ stage: 'welcome', job: null, picked: null, live: checked(status, false) }), 'welcome')
  }
})

test('an import that was started shows how it stands, not what the check found before it', () => {
  const blocked = checked('fda_blocked', true)
  const step = (status: Job['status']) => guideStep({ stage: 'source', job: job({ status }), picked: null, live: blocked })

  assert.equal(step('queued'), 'import_running')
  assert.equal(step('running'), 'import_running')
  assert.equal(step('failed'), 'import_failed')
  // it has ended, and the guide has not yet moved on
  assert.equal(step('completed'), 'import_running')
})

test('an import that stops is a step of its own, so the frame gives it the focus', () => {
  const live = checked('readable', true)
  const steps = [null, job({ status: 'running' }), job({ status: 'failed' })].map((current) =>
    guideStep({ stage: 'source', job: current, picked: null, live }),
  )

  assert.deepEqual(steps, ['import', 'import_running', 'import_failed'])
})

test('when the archive is ready the guide says so', () => {
  const step = guideStep({ stage: 'done', job: job({ status: 'completed' }), picked: null, live: checked('readable', true) })

  assert.equal(step, 'done')
  // whatever else is still known
  for (const status of ['running', 'failed'] as const) {
    const still = guideStep({ stage: 'done', job: job({ status }), picked: copyAt('/copies/chat.db'), live: checked('missing', true) })

    assert.equal(still, 'done', status)
  }
})

test('an import that was started comes before the copy it reads', () => {
  const live = checked('fda_blocked', true)
  const picked = copyAt('/copies/chat.db')

  assert.equal(guideStep({ stage: 'source', job: job({ status: 'running' }), picked, live }), 'import_running')
  assert.equal(guideStep({ stage: 'source', job: job({ status: 'failed' }), picked, live }), 'import_failed')
})

test('the import step shows one of three views, and the others none', () => {
  assert.equal(importView('import'), 'ready')
  assert.equal(importView('import_running'), 'running')
  assert.equal(importView('import_failed'), 'failed')
  for (const step of ['welcome', 'permission', 'no_messages', 'db_problem', 'done'] as const) assert.equal(importView(step), null)
})

test('a copy that was chosen is offered for import, whatever Messages itself allows', () => {
  for (const status of STATUSES) {
    const step = guideStep({ stage: 'source', job: null, picked: copyAt('/copies/chat.db'), live: checked(status, false) })

    assert.equal(step, 'import', status)
  }
})

test('without the copy the guide is back at what Messages itself allows', () => {
  const live = checked('fda_blocked', false)

  assert.equal(guideStep({ stage: 'source', job: null, picked: copyAt('/copies/chat.db'), live }), 'import')
  assert.equal(guideStep({ stage: 'source', job: null, picked: null, live }), 'permission')
})

test('the four marks at the top follow the step', () => {
  const marks: Record<GuideStep, number> = {
    welcome: 0,
    permission: 1,
    no_messages: 1,
    db_problem: 1,
    import: 2,
    import_running: 2,
    import_failed: 2,
    done: 3,
  }

  for (const [step, mark] of Object.entries(marks)) assert.equal(frameStep(step as GuideStep), mark, step)
})

test('an import writes the archive file that the app reads', () => {
  const request = importRequest({ archivePath: '/elsewhere/mine.csv', copyPath: '' })

  assert.deepEqual(request, { messagesPath: '/elsewhere/mine.csv', dbPath: undefined, includeContacts: true })
  // what is sent names the archive and no database
  assert.deepEqual(JSON.parse(JSON.stringify(request)), { messagesPath: '/elsewhere/mine.csv', includeContacts: true })
})

test('with no archive file set, the server writes its own', () => {
  for (const archivePath of ['', '   ']) {
    assert.deepEqual(JSON.parse(JSON.stringify(importRequest({ archivePath, copyPath: '' }))), { includeContacts: true })
  }
})

test('a copy is named only when one was chosen, and with the archive file', () => {
  const request = importRequest({ archivePath: '/elsewhere/mine.csv', copyPath: '/copies/chat.db' })

  assert.deepEqual(request, { messagesPath: '/elsewhere/mine.csv', dbPath: '/copies/chat.db', includeContacts: true })
})

test('the path of the archive is sent without the spaces around it, also with a copy', () => {
  const request = importRequest({ archivePath: '  /elsewhere/mine.csv ', copyPath: '/copies/chat.db' })

  assert.deepEqual(request, { messagesPath: '/elsewhere/mine.csv', dbPath: '/copies/chat.db', includeContacts: true })
})

test('a copy that can be imported has nothing against it', () => {
  assert.equal(whyNotThisCopy('readable'), '')
  assert.equal(whyNotThisCopy('locked'), '')
})

test('a copy that cannot be imported is told why, each in its own words', () => {
  const reasons = (['fda_blocked', 'missing', 'empty', 'invalid', 'error'] as const).map(whyNotThisCopy)

  assert.equal(new Set(reasons).size, 5)
  for (const reason of reasons) assert.ok(reason.length > 0)
  assert.match(whyNotThisCopy('missing'), /no file/)
  assert.match(whyNotThisCopy('empty'), /nothing in it/)
  assert.match(whyNotThisCopy('invalid'), /not a Messages database/)
  // a file that could not be looked at is not said to be another kind of file
  assert.doesNotMatch(whyNotThisCopy('error'), /not a Messages database/)
})

test('an import that the server refused says what the server said', () => {
  const refused = notStartedJob(new Error('An import is already running. Wait until it ends.'), true)

  assert.equal(refused.id, '')
  assert.equal(refused.status, 'failed')
  assert.equal(refused.error, 'An import is already running. Wait until it ends.')
})

test('an import that never reached the server says so in plain words', () => {
  // what a browser raises when the request does not go through
  const unreachable = new TypeError('Failed to fetch')

  assert.equal(notStartedJob(unreachable, true).error, 'Recall did not answer, so the update was not started.')
  assert.equal(notStartedJob(unreachable, false).error, 'Recall did not answer, so the import was not started.')
  assert.equal(notStartedJob('not an error', false).error, 'Recall did not answer, so the import was not started.')
})

test('an import that was never made did not start; one that was made did not finish', () => {
  const never = notStartedJob(new Error('An import is already running. Wait until it ends.'), true)

  assert.equal(failedHeading(never, true), 'The update did not start')
  assert.equal(failedHeading(never, false), 'The import did not start')
  assert.equal(failedHeading(job({ status: 'failed' }), true), 'The update did not finish')
  assert.equal(failedHeading(job({ status: 'failed' }), false), 'The import did not finish')
})

test('an update that was interrupted is called an update, and says what to do', () => {
  assert.equal(interruptedReason(false), 'The import was interrupted.')
  assert.match(interruptedReason(true), /^The update was interrupted\./)
  assert.doesNotMatch(interruptedReason(true), /import/)
})

test('the first words are shown after the first setup, and not after an update', () => {
  const result = { messages: 12, conversations: 2 }

  assert.equal(revealsFirstWords(false, result), true)
  assert.equal(revealsFirstWords(true, result), false)
  // an archive that was already there brought nothing in
  assert.equal(revealsFirstWords(false, null), false)
})

test('a step starts with nothing told at it', () => {
  assert.deepEqual(noNotes('permission'), { step: 'permission', pickError: '', returnedFromSettings: false, sameAgain: false })
})

test('what is told at a step is kept with what was told there before', () => {
  const first = noted(noNotes('permission'), 'permission', { returnedFromSettings: true })
  const second = noted(first, 'permission', { pickError: 'That file could not be read.' })

  assert.deepEqual(second, {
    step: 'permission',
    pickError: 'That file could not be read.',
    returnedFromSettings: true,
    sameAgain: false,
  })
  // the first notes are not changed
  assert.equal(first.pickError, '')
})

test('what is told for a step that is no longer showing is dropped', () => {
  const showing = noNotes('no_messages')

  // an answer that comes late, for the step the reader has left
  assert.equal(noted(showing, 'permission', { pickError: 'That file could not be read.' }), showing)
})

test('a second check found the same only when it got an answer and the answer is the same', () => {
  assert.equal(foundTheSame('missing', checked('missing', false)), true)
  // another kind of nothing is not the same thing again
  assert.equal(foundTheSame('missing', checked('empty', false)), false)
  assert.equal(foundTheSame('missing', checked('readable', false)), false)
  // a check that got no answer found nothing
  assert.equal(foundTheSame('missing', null), false)
})

