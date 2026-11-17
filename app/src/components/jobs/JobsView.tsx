import { useEffect, useMemo } from 'react'
import { parseDate, relativeTime } from '../../lib/format'
import { isActiveJob, jobTitle, lastLog, mergeJobs } from '../../lib/jobs'
import type { Contact, Job } from '../../types'
import { Truncate } from '../ui/Truncate'

type JobsViewProps = {
  jobs: Job[]
  // the report this session started and is following
  runningJob: Job | null
  contacts: Contact[]
  onRefresh: () => void
}

// while anything is running the list is read again this often
const REFRESH_MS = 2000

const STATES: Record<Job['status'], { word: string; tone: string }> = {
  queued: { word: 'Waiting', tone: 'busy' },
  running: { word: 'Running', tone: 'busy' },
  completed: { word: 'Done', tone: 'ok' },
  failed: { word: 'Failed', tone: 'bad' },
}

// what a job is doing now, or why it stopped
function detailOf(job: Job) {
  if (job.status === 'failed') return job.error || lastLog(job.logs) || 'It stopped without saying why'
  if (!isActiveJob(job)) return ''
  if (job.progress) return `Step ${job.progress.step} of ${job.progress.of}: ${job.progress.label}`
  return lastLog(job.logs)
}

// The work this session has asked for: reports, imports, the index. It keeps
// itself current while anything is running.
export function JobsView({ jobs, runningJob, contacts, onRefresh }: JobsViewProps) {
  const listed = useMemo(() => mergeJobs(jobs, runningJob), [jobs, runningJob])
  const anyActive = listed.some(isActiveJob)

  useEffect(() => {
    onRefresh()
  }, [onRefresh])

  useEffect(() => {
    if (!anyActive) return
    const timer = window.setInterval(onRefresh, REFRESH_MS)
    return () => window.clearInterval(timer)
  }, [anyActive, onRefresh])

  if (!listed.length) {
    return (
      <div className="view">
        <div className="pane-empty arrive-each">
          <p>No jobs yet</p>
        </div>
      </div>
    )
  }

  return (
    <div className="view">
      <div className="page wide arrive">
        <div className="card">
          <ul className="rows" role="list">
            {listed.map((job) => {
              const state = STATES[job.status]
              const detail = detailOf(job)
              const updated = parseDate(job.updatedAt)
              return (
                <li key={job.id}>
                  <div className="row job">
                    <span className="job-main">
                      <Truncate className="job-title">{jobTitle(job, contacts)}</Truncate>
                      {detail ? <span className={job.status === 'failed' ? 'job-detail bad' : 'job-detail'}>{detail}</span> : null}
                    </span>
                    <span className={`status ${state.tone}`}>
                      {isActiveJob(job) ? <span className="spinner" /> : null}
                      {state.word}
                    </span>
                    {updated ? (
                      <time className="job-when" dateTime={updated.toISOString()}>
                        {relativeTime(updated.valueOf())}
                      </time>
                    ) : (
                      <span className="job-when" />
                    )}
                  </div>
                </li>
              )
            })}
          </ul>
        </div>
      </div>
    </div>
  )
}
