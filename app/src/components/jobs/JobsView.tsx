import { useEffect, useMemo, useRef, useState } from 'react'
import { isActiveJob, mergeJobs } from '../../lib/jobs'
import type { Contact, Job, ReportFile } from '../../types'
import { JobDetail } from './JobDetail'
import { JobList } from './JobList'

type JobsViewProps = {
  jobs: Job[]
  // the report this session started and is following
  runningJob: Job | null
  contacts: Contact[]
  reports: ReportFile[]
  onOpenReport: (report: ReportFile) => void
  onRefresh: () => void
}

// while anything is running the list is read again this often
const REFRESH_MS = 2000

// the report a finished job wrote, while it is still among the saved reports
function reportOf(job: Job, reports: ReportFile[]) {
  const path = job.status === 'completed' ? job.result?.reportPath : undefined
  return path ? (reports.find((report) => report.path === path) ?? null) : null
}

// The work this session has asked for: reports, imports, the index. The jobs
// are on the left and the chosen one is on the right: how it stands and what
// it wrote. In a narrow window the two take turns. It keeps itself current
// while anything is running.
export function JobsView({ jobs, runningJob, contacts, reports, onOpenReport, onRefresh }: JobsViewProps) {
  const listed = useMemo(() => mergeJobs(jobs, runningJob), [jobs, runningJob])
  const anyActive = listed.some(isActiveJob)
  // the job the reader chose; until one is chosen, and when it is gone, the newest
  const [chosenId, setChosenId] = useState('')
  // in a narrow window only one of the two panes shows
  const [pane, setPane] = useState<'list' | 'detail'>('list')
  const split = useRef<HTMLDivElement>(null)

  useEffect(() => {
    onRefresh()
  }, [onRefresh])

  useEffect(() => {
    if (!anyActive) return
    const timer = window.setInterval(onRefresh, REFRESH_MS)
    return () => window.clearInterval(timer)
  }, [anyActive, onRefresh])

  // In a narrow window the pane a press came from is gone once the other
  // takes its place, and the focus with it. It goes to the way back, or back
  // to the row of the job that was open. Where both panes show, nothing was
  // lost and nothing moves.
  useEffect(() => {
    const held = document.activeElement
    const lost = !(held instanceof HTMLElement) || held === document.body || held.offsetParent === null
    const next = split.current?.querySelector<HTMLElement>(pane === 'detail' ? '.pane-back' : '.job-row[aria-current]')
    if (lost && next && next.offsetParent !== null) next.focus()
  }, [pane])

  if (!listed.length) {
    return (
      <div className="view">
        <div className="pane-empty arrive-each">
          <p>No jobs yet</p>
        </div>
      </div>
    )
  }

  const open = listed.find((job) => job.id === chosenId) ?? listed[0]

  return (
    <div ref={split} className="split jobs-split" data-pane={pane}>
      <JobList
        jobs={listed}
        contacts={contacts}
        openId={open.id}
        onShow={(job) => setChosenId(job.id)}
        onOpen={(job) => {
          setChosenId(job.id)
          setPane('detail')
        }}
      />
      <div className="divider" />
      <JobDetail
        // another job starts at its own top, and arrives
        key={open.id}
        job={open}
        contacts={contacts}
        report={reportOf(open, reports)}
        onOpenReport={onOpenReport}
        onShowList={() => setPane('list')}
      />
    </div>
  )
}
