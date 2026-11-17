// The archive can be in the middle of being loaded when the app asks for it:
// the server then answers "archive_building" and names the job that does the
// load. These are the rules of the wait, with no screen in them.
import type { Job } from '../types'
import { ApiError } from './apiError.ts'
import { lastLog } from './jobs.ts'

const ARCHIVE_BUILDING = 'archive_building'
const NO_LABEL = 'Loading your messages'

export const BUILD_FAILED = 'Recall could not get your archive ready.'
export const BUILD_LOST = 'Recall stopped answering while it got your archive ready.'

function isJob(value: unknown): value is Job {
  const job = value as Partial<Job> | null
  return Boolean(job && typeof job === 'object' && typeof job.id === 'string' && job.id && typeof job.status === 'string')
}

// The job that loads the archive, from the error of a request that could not
// read the archive yet. Null for any other error, and for an answer that
// names no job: there is nothing to ask about then.
export function buildingJob(error: unknown): Job | null {
  if (!(error instanceof ApiError) || error.code !== ARCHIVE_BUILDING) return null
  const job = (error.body as { job?: unknown } | null | undefined)?.job
  return isJob(job) ? job : null
}

// For the catch of a request: the job in place of the answer when the
// archive is loading. Any other error is thrown again.
export function orBuilding(error: unknown): { building: Job } {
  const building = buildingJob(error)
  if (!building) throw error
  return { building }
}

// how the wait stands after an answer about the job
export function buildEnd(job: Job): 'running' | 'ready' | 'failed' {
  if (job.status === 'completed') return 'ready'
  return job.status === 'failed' ? 'failed' : 'running'
}

// why a load failed, in the words of the server where it gave some
export function buildFailure(job: Job) {
  return job.error || BUILD_FAILED
}

// What the waiting window says: what the load is doing now, and the line it
// last wrote when that says something more.
export function buildView(job: Job): { label: string; said: string } {
  const label = job.progress?.label?.trim() || NO_LABEL
  const said = lastLog(job.logs)
  return { label, said: said === label ? '' : said }
}
