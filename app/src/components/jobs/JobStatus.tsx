import { isActiveJob } from '../../lib/jobs'
import type { Job } from '../../types'

const STATES: Record<Job['status'], { word: string; tone: string }> = {
  queued: { word: 'Waiting', tone: 'busy' },
  running: { word: 'Running', tone: 'busy' },
  completed: { word: 'Done', tone: 'ok' },
  failed: { word: 'Failed', tone: 'bad' },
}

// How a job stands, as a mark and a word; a spinner is the mark while it works.
export function JobStatus({ job }: { job: Job }) {
  const state = STATES[job.status]
  return (
    <span className={`status ${state.tone}`}>
      {isActiveJob(job) ? <span className="spinner" /> : null}
      {state.word}
    </span>
  )
}
