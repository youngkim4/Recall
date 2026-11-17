// The rules of the setup guide, with no screen in them: which step shows for
// what was last found, what an import is asked for, and what is said when one
// does not start or does not end.
import type { Job, SetupDbStatus, SetupStatus } from '../types'

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

// which of the four marks at the top a step lights
export function frameStep(step: GuideStep) {
  if (step === 'welcome') return 0
  if (step === 'done') return 3
  return step === 'import' || step === 'import_running' || step === 'import_failed' ? 2 : 1
}

// What the guide asks the server for when it starts an import. The archive
// file is the one that the app reads; with none set, the server writes its
// own. A database is named only when a copy was chosen in this guide.
export function importRequest(input: { archivePath: string; copyPath: string }) {
  return {
    messagesPath: input.archivePath.trim() || undefined,
    dbPath: input.copyPath || undefined,
    includeContacts: true,
  }
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
