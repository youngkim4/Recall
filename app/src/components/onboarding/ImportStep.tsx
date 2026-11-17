import { formatNumber, plural } from '../../lib/format'
import { isActiveJob, lastLog } from '../../lib/jobs'
import type { Job, SetupStatus } from '../../types'
import { CheckIcon } from '../Icons'

type ImportStepProps = {
  status: SetupStatus
  // the import, once it has been started
  job: Job | null
  rerun: boolean
  onStart: () => void
  onRetry: () => void
  onBack: () => void
}

const PHASES = ['Reading the Messages database', 'Matching names from Contacts', 'Getting the archive ready']

// the size of what is about to be read, rounded the way the quick count is
function sizeOf(db: SetupStatus['db']) {
  if (!db.approxMessages) return ''
  const messages =
    db.approxMessages >= 1000 ? `${formatNumber(Math.round(db.approxMessages / 1000) * 1000)} messages` : plural(db.approxMessages, 'message')
  return [
    `About ${messages}`,
    db.conversations ? `in ${plural(db.conversations, 'conversation')}` : '',
    db.firstYear ? `since ${db.firstYear}` : '',
  ]
    .filter(Boolean)
    .join(' ')
}

// Before, during and after the import: what will be read, how far it has got,
// or why it stopped.
export function ImportStep({ status, job, rerun, onStart, onRetry, onBack }: ImportStepProps) {
  if (job?.status === 'failed') {
    return (
      <>
        <h1>{rerun ? 'The update did not finish' : 'The import did not finish'}</h1>
        {job.error ? (
          <p className="notice bad" role="alert">
            {job.error}
          </p>
        ) : null}
        <p className="setup-lead">Messages and its database are as they were.</p>
        <div className="setup-actions">
          <button type="button" className="btn primary lg" onClick={onRetry}>
            Try again
          </button>
          <button type="button" className="btn quiet lg" onClick={onBack}>
            {rerun ? 'Not now' : 'Back'}
          </button>
        </div>
      </>
    )
  }

  if (isActiveJob(job)) {
    const current = job?.progress?.step || 1
    const said = lastLog(job?.logs)
    return (
      <>
        <h1>{rerun ? 'Updating your archive' : 'Building your archive'}</h1>
        <ol className="setup-phases" role="list" aria-label="Progress">
          {PHASES.map((phase, index) => {
            const place = index + 1
            const state = place < current ? 'done' : place === current ? 'now' : undefined
            return (
              <li key={phase} data-state={state} aria-current={state === 'now' ? 'step' : undefined}>
                <span className="setup-phase-mark" aria-hidden="true">
                  {state === 'done' ? <CheckIcon /> : state === 'now' ? <span className="spinner" /> : place}
                </span>
                {phase}
              </li>
            )
          })}
        </ol>
        {/* the line the import last wrote, so a long step is seen to be moving */}
        <p className="setup-note setup-said" role="status">
          {said}
        </p>
      </>
    )
  }

  const size = sizeOf(status.db)
  // a copy is read only because it was chosen in this guide, and the reader is told what that does
  const fromCopy = status.db.kind === 'copy'
  return (
    <>
      <h1>{rerun ? 'Update your archive' : 'Build your archive'}</h1>
      {size ? <p className="setup-size">{size}</p> : null}
      <p className="setup-lead">
        {fromCopy
          ? 'Recall reads the copy you chose. The archive then holds what that copy holds and nothing else. The copy is not changed.'
          : rerun
            ? 'Recall reads Messages again. The archive then holds what Messages holds now, so a message deleted there leaves the archive too. Messages and its database are not changed.'
            : 'Recall copies your message history into its own file on this Mac. Messages and its database are not changed.'}
      </p>
      <div className="setup-actions">
        <button type="button" className="btn primary lg" onClick={onStart}>
          {rerun ? 'Update archive' : 'Import messages'}
        </button>
        <button type="button" className="btn quiet lg" onClick={onBack}>
          {rerun ? 'Not now' : 'Back'}
        </button>
      </div>
    </>
  )
}
