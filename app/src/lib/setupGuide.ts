// The rules of the setup guide, with no screen in them: which step shows for
// what was last found, what an import is asked for, and what is said when one
// does not start or does not end.
import type { Job, SetupDbStatus, SetupImportResult, SetupStatus } from '../types'

// How far the reader has gone, apart from what the database allows. A first
// setup opens on its welcome; an update opens on what there is to do.
export type GuideStage = 'welcome' | 'source' | 'done'

export type GuideStep =
  | 'welcome'
  | 'permission'
  | 'no_messages'
  | 'db_problem'
  // the import, before it is started
  | 'import'
  | 'import_running'
  | 'import_failed'
  | 'done'

// A database that the server will try. A busy one is among them: Messages may
// only be in the middle of a write, and the server tries once more.
export function canImport(status: SetupDbStatus) {
  return status === 'readable' || status === 'locked'
}

// The step for what a check of the database found.
export function stepForDatabase(status: SetupDbStatus): GuideStep {
  if (canImport(status)) return 'import'
  if (status === 'fda_blocked') return 'permission'
  if (status === 'missing' || status === 'empty') return 'no_messages'
  return 'db_problem'
}

// The step that shows. It is worked out again from what is known each time,
// so it always fits the last check: no step is kept from an earlier one.
// `live` is the check of Messages itself, and `picked` the check of a copy
// chosen in this guide, if there is one. What the server says of setup
// (`state`) does not choose the step: with an archive file there it says
// "ready" whatever the database is.
export function guideStep(input: {
  stage: GuideStage
  job: Job | null
  picked: SetupStatus | null
  live: SetupStatus
}): GuideStep {
  if (input.stage === 'welcome') return 'welcome'
  if (input.stage === 'done') return 'done'
  if (input.job) return input.job.status === 'failed' ? 'import_failed' : 'import_running'
  return input.picked ? 'import' : stepForDatabase(input.live.db.status)
}

// What the import step shows: the import before it is started, while it
// runs, or after it stopped. Another step shows none of them.
export type ImportView = 'ready' | 'running' | 'failed'

export function importView(step: GuideStep): ImportView | null {
  if (step === 'import') return 'ready'
  if (step === 'import_running') return 'running'
  return step === 'import_failed' ? 'failed' : null
}

// which of the four marks at the top a step lights
export function frameStep(step: GuideStep) {
  if (step === 'welcome') return 0
  if (step === 'done') return 3
  return importView(step) ? 2 : 1
}

// What the reader was told at a step. It belongs to that step: when another
// step shows, it is gone, and it does not come back with the step.
export type StepNotes = {
  step: GuideStep
  // why the copy that was tried cannot be used
  pickError: string
  // the reader went to System Settings and came back to the window
  returnedFromSettings: boolean
  // "Check again" found the same thing as before
  sameAgain: boolean
}

export function noNotes(step: GuideStep): StepNotes {
  return { step, pickError: '', returnedFromSettings: false, sameAgain: false }
}

// The notes with a change that was made at `step`. A change for a step that
// is no longer showing is dropped: it is the late answer to something the
// reader has left.
export function noted(notes: StepNotes, step: GuideStep, change: Partial<Omit<StepNotes, 'step'>>): StepNotes {
  return notes.step === step ? { ...notes, ...change } : notes
}

// Whether a second check found what the first one found. A check that got no
// answer found nothing.
export function foundTheSame(before: SetupDbStatus, next: SetupStatus | null) {
  return next !== null && next.db.status === before
}

// Whether the guide is an update: the archive that the app reads holds
// messages. The server says so, and it uses the same test to decide whether
// a copy can be named (holds_messages), so the guide never offers what the
// server then refuses.
export function isUpdate(status: SetupStatus) {
  return status.export.holdsMessages === true
}

// A copy can be named in a first setup only. An update reads Messages on
// this Mac and nothing else, so its guide offers no other source. The server
// holds the same rule (database_for).
export function offersCopy(rerun: boolean) {
  return !rerun
}

// What the guide asks the server for when it starts an import. The archive
// file is the one that the app reads; with none set, the server writes its
// own. A database is named only when a copy was chosen in a first setup.
// allowRemoval is sent only when the reader said to go on past the limit on
// removed messages; without the key the server keeps the limit.
export function importRequest(input: { archivePath: string; copyPath: string; rerun: boolean; allowRemoval?: boolean }): {
  messagesPath: string | undefined
  dbPath: string | undefined
  includeContacts: boolean
  allowRemoval?: true
} {
  const request = {
    messagesPath: input.archivePath.trim() || undefined,
    dbPath: (offersCopy(input.rerun) && input.copyPath) || undefined,
    includeContacts: true,
  }
  return input.allowRemoval ? { ...request, allowRemoval: true } : request
}

// Why a copy cannot be used, or '' when it can.
export function whyNotThisCopy(status: SetupDbStatus) {
  if (canImport(status)) return ''
  if (status === 'missing') return 'There is no file at that path.'
  if (status === 'empty') return 'That database has nothing in it.'
  if (status === 'fda_blocked') {
    return 'macOS blocked that folder. Allow it in the prompt macOS showed, or move the file to your home folder and choose it again.'
  }
  if (status === 'error') return 'That file could not be read.'
  return 'That file is not a Messages database. Choose chat.db, not chat.db-wal or chat.db-shm.'
}

const called = (rerun: boolean) => (rerun ? 'update' : 'import')

// An import that was asked for and not made, as a job that failed. It has no
// id: the server holds no job for it. What the server said is kept; a request
// that did not go through has no words of the server to show.
export function notStartedJob(error: unknown, rerun: boolean): Job {
  const said = error instanceof Error && !(error instanceof TypeError) ? error.message : ''
  return {
    id: '',
    action: 'setup_import',
    status: 'failed',
    createdAt: '',
    updatedAt: '',
    logs: [],
    error: said || `Recall did not answer, so the ${called(rerun)} was not started.`,
  }
}

export function failedHeading(job: Pick<Job, 'id'>, rerun: boolean) {
  return `The ${called(rerun)} did not ${job.id ? 'finish' : 'start'}`
}

// the code of an update that Messages lacks too many of the stored messages for
const REMOVAL_LIMIT = 'removal_limit'

// What the failed step shows: the heading, what the server said, one plain
// line under it, and the way on.
export type FailedImport = {
  heading: string
  // the sentence of the server; '' when it gave none
  reason: string
  note: string
  // 'retry' looks at Messages again before the next try. 'allow_removal'
  // starts the same request again and lets the messages go.
  action: 'retry' | 'allow_removal'
  actionLabel: string
}

// An update that would take more messages out of the archive than the limit
// allows changes nothing and fails with its own code. The reader is told
// what going on does, and can go on. Every other failure offers another try.
export function failedImport(job: Pick<Job, 'id' | 'error' | 'code'> | null, rerun: boolean): FailedImport {
  const overLimit = job?.code === REMOVAL_LIMIT
  return {
    heading: failedHeading({ id: job?.id ?? '' }, rerun),
    reason: job?.error || '',
    note: overLimit ? 'Updating removes those messages from the archive too.' : 'Messages and its database are as they were.',
    action: overLimit ? 'allow_removal' : 'retry',
    actionLabel: overLimit ? 'Update anyway' : 'Try again',
  }
}

// What is said when the server stops answering about an import. After an
// update the archive file says nothing about this run, so the reader is told
// how to be sure.
export function interruptedReason(rerun: boolean) {
  return rerun
    ? 'The update was interrupted. Start it again to be sure that the archive is up to date.'
    : 'The import was interrupted.'
}

// The first words are shown once, after the first setup. An update ends
// without them, and so does a setup that found the archive already there.
export function revealsFirstWords(rerun: boolean, result: SetupImportResult | null) {
  return !rerun && result !== null
}
