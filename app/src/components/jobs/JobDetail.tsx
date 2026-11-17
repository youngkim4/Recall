import { useMemo } from 'react'
import { formatDateTime } from '../../lib/format'
import { jobTimes, logLines, standingOf, stepLine, stepStates, whatItRead } from '../../lib/jobFacts'
import { isActiveJob, jobTitle } from '../../lib/jobs'
import type { Contact, Job, ReportFile } from '../../types'
import { ChevronLeftIcon } from '../Icons'
import { JobLog } from './JobLog'
import { JobStatus } from './JobStatus'

type JobDetailProps = {
  job: Job
  contacts: Contact[]
  // the report this job wrote, while it is among the saved reports
  report: ReportFile | null
  onOpenReport: (report: ReportFile) => void
  // in a narrow window, the way back to the list
  onShowList: () => void
}

// One job, filling its pane: its name and how it stands, the step it is on
// while it runs, how it ended, and what it wrote, in a card that takes the
// height that is left.
export function JobDetail({ job, contacts, report, onOpenReport, onShowList }: JobDetailProps) {
  const title = jobTitle(job, contacts)
  const active = isActiveJob(job)
  const lines = useMemo(() => logLines(job.logs), [job.logs])
  const figures = job.status === 'completed' ? whatItRead(job.result) : []

  return (
    <section className="pane pane-detail" aria-label={title}>
      <div className="detail job-detail arrive-each">
        <button type="button" className="pane-back btn quiet" onClick={onShowList}>
          <ChevronLeftIcon />
          Jobs
        </button>

        <header>
          <div className="section-head">
            <h2 className="detail-title">{title}</h2>
            <JobStatus job={job} />
            {report ? (
              <span className="section-links">
                <button type="button" className="link" onClick={() => onOpenReport(report)}>
                  Open report
                </button>
              </span>
            ) : null}
          </div>
          <p className="detail-sub job-times">
            {jobTimes(job, formatDateTime).map((time) => (
              <span key={time}>{time}</span>
            ))}
          </p>
        </header>

        {active && job.progress ? <JobSteps progress={job.progress} /> : null}

        {job.status === 'failed' ? (
          <p className="notice bad" role="alert">
            {standingOf(job)}
          </p>
        ) : null}

        {figures.length ? (
          <dl className="job-figures">
            {figures.map((figure) => (
              <div key={figure.label}>
                <dt>{figure.label}</dt>
                <dd>{figure.value}</dd>
              </div>
            ))}
          </dl>
        ) : null}

        {job.status === 'completed' && job.result?.contactsSkipped ? (
          <p className="notice">Names from Contacts were not read</p>
        ) : null}

        <JobLog lines={lines} following={active} />
      </div>
    </section>
  )
}

// The step a running job is on, in words, over one mark for each of its steps.
function JobSteps({ progress }: { progress: NonNullable<Job['progress']> }) {
  const words = stepLine(progress)
  return (
    <div className="job-progress">
      <p>{words}</p>
      <div
        className="job-steps"
        role="progressbar"
        aria-label="Steps"
        aria-valuemin={0}
        aria-valuemax={progress.of}
        aria-valuenow={progress.step}
        aria-valuetext={words}
      >
        {stepStates(progress).map((state, index) => (
          <i key={index} data-step={state} />
        ))}
      </div>
    </div>
  )
}
