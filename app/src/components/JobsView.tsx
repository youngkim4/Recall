import type { Job } from '../types'
import { formatNumber, shortDateTime } from '../lib/format'

type JobsViewProps = {
  jobs: Job[]
  runningJob: Job | null
  utilityJob: Job | null
  loading: boolean
  onRefresh: () => Promise<void>
}

export function JobsView({ jobs, runningJob, utilityJob, loading, onRefresh }: JobsViewProps) {
  const merged = mergeJobs(jobs, runningJob, utilityJob)
  const active = merged.filter((job) => job.status === 'queued' || job.status === 'running')

  return (
    <section className="jobs-view" aria-label="Jobs">
      <div className="settings-header">
        <div>
          <h2>Jobs</h2>
          <p>Report runs, message exports, and maintenance work from this session.</p>
        </div>
        <button type="button" className="button ghost" disabled={loading} onClick={() => void onRefresh()}>
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      <div className="metrics-grid job-summary">
        <div className="metric">
          <span>Total</span>
          <strong>{formatNumber(merged.length)}</strong>
        </div>
        <div className="metric">
          <span>Active</span>
          <strong>{formatNumber(active.length)}</strong>
        </div>
        <div className="metric">
          <span>Reports</span>
          <strong>{formatNumber(merged.filter((job) => job.action === 'analyze').length)}</strong>
        </div>
        <div className="metric">
          <span>Exports</span>
          <strong>{formatNumber(merged.filter((job) => job.action === 'export').length)}</strong>
        </div>
      </div>

      <div className="job-list">
        {merged.map((job) => (
          <article key={job.id} className={`job-row ${job.status}`}>
            <div>
              <span className="eyebrow">{job.action}</span>
              <strong>{job.status}</strong>
              <p>{job.error || lastLog(job.logs) || 'No logs yet.'}</p>
            </div>
            <time>{shortDateTime(job.updatedAt)}</time>
          </article>
        ))}
        {!merged.length ? (
          <div className="empty-state tall">
            <strong>No jobs yet.</strong>
            <span>Run a report or refresh the message export to populate history.</span>
          </div>
        ) : null}
      </div>
    </section>
  )
}

function mergeJobs(jobs: Job[], runningJob: Job | null, utilityJob: Job | null) {
  const byId = new Map<string, Job>()
  for (const job of jobs) byId.set(job.id, job)
  if (runningJob) byId.set(runningJob.id, runningJob)
  if (utilityJob) byId.set(utilityJob.id, utilityJob)
  return [...byId.values()].sort((left, right) => String(right.updatedAt).localeCompare(String(left.updatedAt)))
}

function lastLog(logs: Array<string | { message?: string }> = []) {
  for (let index = logs.length - 1; index >= 0; index -= 1) {
    const entry = logs[index]
    const message = typeof entry === 'string' ? entry : entry?.message || ''
    if (message && !message.startsWith('127.0.0.1 - -')) return message
  }
  return ''
}
