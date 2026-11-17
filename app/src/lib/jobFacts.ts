// What a job says about itself: where it stands, the steps it goes through,
// the lines it wrote, and what it read.
import type { Job } from '../types'
import { formatNumber } from './format.ts'
import { isActiveJob, lastLog } from './jobs.ts'

export type LogLine = {
  // when the line was written, as the server gave it; '' when it did not say
  time: string
  words: string
}

export type Figure = { label: string; value: string }
export type StepState = 'done' | 'now' | 'ahead'

// the server's own record of a request it answered, which no reader wants
const REQUEST = '127.0.0.1 - -'
// more steps than this cannot be drawn as marks in a row
const MOST_STEPS = 24
// a reader this close to the end of a log is at its end: about a line
const NEAR_END = 24

// What a job wrote, oldest first: the lines a person would read, each with
// the time it was written where the server gave one.
export function logLines(logs: Job['logs'] = []): LogLine[] {
  return logs
    .map((entry) =>
      typeof entry === 'string'
        ? { time: '', words: entry.trim() }
        : { time: entry?.time ?? '', words: (entry?.message ?? '').trim() },
    )
    .filter((line) => line.words && !line.words.startsWith(REQUEST))
}

export function stepLine(progress: NonNullable<Job['progress']>) {
  return `Step ${progress.step} of ${progress.of}: ${progress.label}`
}

// What a job is doing now, or why it stopped. One that ended well, or has
// not begun, has nothing more to say than its status.
export function standingOf(job: Job) {
  if (job.status === 'failed') return job.error || lastLog(job.logs) || 'It stopped without saying why'
  if (!isActiveJob(job)) return ''
  return job.progress ? stepLine(job.progress) : lastLog(job.logs)
}

// what the second time of a job means, by how the job stands
const ENDED: Partial<Record<Job['status'], string>> = {
  running: 'Last change',
  completed: 'Finished',
  failed: 'Stopped',
}

// When a job began, and when it ended or last changed, as short lines. `say`
// writes a time out. A job that waits has only the time it was asked for; a
// time the job does not have is not written.
export function jobTimes(job: Job, say: (time: string) => string) {
  const began = job.createdAt ? `${job.status === 'queued' ? 'Asked for' : 'Started'} ${say(job.createdAt)}` : ''
  const word = ENDED[job.status]
  const ended = word && job.updatedAt ? `${word} ${say(job.updatedAt)}` : ''
  return [began, ended].filter(Boolean)
}

// One mark for each step of a job: the ones behind the step in hand, that
// step, and the ones ahead. A step outside its count is held inside it.
export function stepStates(progress: { step: number; of: number }): StepState[] {
  const count = Math.floor(progress.of)
  if (!(count >= 1) || count > MOST_STEPS) return []
  const now = Math.max(1, Math.min(count, Math.floor(progress.step) || 1))
  return Array.from({ length: count }, (_, index) => (index + 1 < now ? 'done' : index + 1 === now ? 'now' : 'ahead'))
}

// The years a job read: one year, or the first and the last.
function yearsOf(first?: number | null, last?: number | null) {
  if (!first || !last) return ''
  return first === last ? String(first) : `${first}–${last}`
}

// What a finished job read from Messages, as figures. A figure the job did
// not give is not shown; a job that gave none has no figures.
export function whatItRead(result: Job['result']): Figure[] {
  if (!result) return []
  const years = yearsOf(result.firstYear, result.lastYear)
  const figures: Array<Figure | null> = [
    typeof result.messages === 'number' ? { label: 'Messages', value: formatNumber(result.messages) } : null,
    typeof result.conversations === 'number' ? { label: 'Conversations', value: formatNumber(result.conversations) } : null,
    years ? { label: 'Years', value: years } : null,
    // messages that iCloud had taken off this Mac, put back from the saved copy
    result.recovered ? { label: 'Restored', value: formatNumber(result.recovered) } : null,
  ]
  return figures.filter((figure): figure is Figure => figure !== null)
}

// Whether a reader is at the end of a list that scrolls, give or take a line.
export function atEnd(box: { scrollHeight: number; scrollTop: number; clientHeight: number }) {
  return box.scrollHeight - box.scrollTop - box.clientHeight <= NEAR_END
}
