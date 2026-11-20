// Getting chat ready, with no screen in it: what state it is in, what it costs
// at most, and what to say of a job on its way. The shapes are those of
// /api/first-run, /api/key and the first_run job (recall_services/README.md,
// "First run").
import type { FirstRunStatus, FirstRunStepName, Job, KeyState } from '../types.ts'
import { isActiveJob } from './jobs.ts'

// the three things chat needs, as the reader is told of them, in the order they are built
export const STEP_LABELS: Record<FirstRunStepName, string> = {
  meaning: 'Search by meaning',
  memory: 'Read every conversation',
  people: 'A profile of each person',
}

// The estimate is an upper bound, so it is rounded up: "at most" must hold.
export function costCeiling(usd: number) {
  return Math.ceil(Math.round(usd * 1e6) / 1e4) / 100
}

export type ChatPhase = 'unknown' | 'no_messages' | 'not_ready' | 'running' | 'failed' | 'ready'

// A job on its way or one that failed says more than the status, which is
// asked again only when a job has ended.
export function chatPhase(status: FirstRunStatus | null, job: Job | null): ChatPhase {
  if (job && isActiveJob(job)) return 'running'
  if (job?.status === 'failed') return 'failed'
  if (!status) return 'unknown'
  if (!status.archive.holdsMessages) return 'no_messages'
  return status.ready ? 'ready' : 'not_ready'
}

// Why a step that is not ready has no cost: its own sentence, not "$0".
export function stepProblems(status: FirstRunStatus): string[] {
  return status.steps.flatMap((step) => (step.state !== 'ready' && step.error ? [step.error] : []))
}

// "Reading your conversations: 40 of 2,300", or the label alone
export function progressLine(job: Job) {
  const progress = job.progress
  if (!progress) return ''
  const { done, total } = progress
  return Number.isFinite(done) && Number.isFinite(total) && total
    ? `${progress.label}: ${new Intl.NumberFormat().format(done ?? 0)} of ${new Intl.NumberFormat().format(total ?? 0)}`
    : progress.label
}

export function keyLine(key: Pick<KeyState, 'present' | 'source'>) {
  if (!key.present) return 'None'
  return key.source === 'environment' ? 'Set where Recall starts' : 'Saved'
}
