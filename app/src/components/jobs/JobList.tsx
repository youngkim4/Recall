import type { FocusEvent } from 'react'
import { parseDate, relativeTime } from '../../lib/format'
import { standingOf } from '../../lib/jobFacts'
import { jobTitle } from '../../lib/jobs'
import { rovingKeyDown } from '../../lib/roving'
import type { Contact, Job } from '../../types'
import { Truncate } from '../ui/Truncate'
import { JobStatus } from './JobStatus'

type JobListProps = {
  // newest first
  jobs: Job[]
  contacts: Contact[]
  // the job whose detail is open
  openId: string
  // the keyboard came to a row: its job is shown beside the list
  onShow: (job: Job) => void
  // a row was pressed: its job is shown, and in a narrow window the list gives way to it
  onOpen: (job: Job) => void
}

// when a job last changed, for a reader and for a machine
function lastChange(job: Job) {
  const date = parseDate(job.updatedAt || job.createdAt)
  return date ? { stamp: date.toISOString(), words: relativeTime(date.valueOf()) } : null
}

// The work asked for, newest first: each job's name and when it last
// changed, and under them how it stands. One stop for Tab; the arrow keys
// move through it, and the job beside the list follows the row they are on.
export function JobList({ jobs, contacts, openId, onShow, onOpen }: JobListProps) {
  function handleFocus(event: FocusEvent<HTMLUListElement>) {
    const rows = [...event.currentTarget.querySelectorAll<HTMLElement>('[data-roving]')]
    const job = jobs[rows.indexOf(event.target as HTMLElement)]
    if (job) onShow(job)
  }

  return (
    <aside className="pane appear" aria-label="Jobs">
      <div className="pane-scroll job-list">
        <ul className="rows" role="list" onKeyDown={rovingKeyDown} onFocus={handleFocus}>
          {jobs.map((job) => {
            const open = job.id === openId
            const standing = standingOf(job)
            const changed = lastChange(job)
            return (
              <li key={job.id}>
                <button
                  type="button"
                  className={open ? 'row job-row is-active' : 'row job-row'}
                  aria-current={open ? 'true' : undefined}
                  data-roving
                  tabIndex={open ? 0 : -1}
                  onClick={() => onOpen(job)}
                >
                  <Truncate className="job-name">{jobTitle(job, contacts)}</Truncate>
                  {changed ? (
                    <time className="job-when" dateTime={changed.stamp}>
                      {changed.words}
                    </time>
                  ) : null}
                  <span className="job-standing">
                    <JobStatus job={job} />
                    {standing ? (
                      <span className={job.status === 'failed' ? 'job-doing bad' : 'job-doing'}>{standing}</span>
                    ) : null}
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
      </div>
    </aside>
  )
}
