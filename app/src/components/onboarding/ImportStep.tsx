import { formatNumber, plural } from '../../lib/format'
import { lastLog } from '../../lib/jobs'
import { failedHeading, type ImportView } from '../../lib/setupGuide'
import type { Job, SetupStatus } from '../../types'
import { ImportProgress } from './ImportProgress'

type ImportStepProps = {
  // before the import is started, while it runs, or after it stopped
  view: ImportView
  // the check of what will be read: Messages itself, or the copy that was chosen
  status: SetupStatus
  // the import, once it has been started
  job: Job | null
  rerun: boolean
  // the request to start is on its way: nothing here can be pressed until it is answered
  starting: boolean
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
export function ImportStep({ view, status, job, rerun, starting, onStart, onRetry, onBack }: ImportStepProps) {
  // a copy is read only because it was chosen in this guide, and the reader is told what that does
  const fromCopy = status.db.kind === 'copy'
  // Back gives a copy up. An update that reads Messages itself has only the app to go back to.
  const wayBack = rerun && !fromCopy ? 'Not now' : 'Back'

  if (view === 'failed') {
    return (
      <>
        <h1>{failedHeading({ id: job?.id ?? '' }, rerun)}</h1>
        {/* the reason says whether the archive was changed, where the server knows */}
        {job?.error ? (
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
            {wayBack}
          </button>
        </div>
      </>
    )
  }

  if (view === 'running') {
    return (
      <>
        <h1>{rerun ? 'Updating your archive' : 'Building your archive'}</h1>
        <ImportProgress phases={PHASES} current={job?.progress?.step || 1} said={lastLog(job?.logs)} />
      </>
    )
  }

  const size = sizeOf(status.db)
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
      {status.db.status === 'locked' ? (
        // a busy database can be imported: the server tries it, and tries once more
        <p className="setup-note">
          {fromCopy ? 'Another app has the copy open now.' : 'Messages is writing to its database now.'} Recall waits a
          moment and tries once more if it is still busy.
        </p>
      ) : null}
      <div className="setup-actions">
        <button type="button" className="btn primary lg" disabled={starting} onClick={onStart}>
          {starting ? <span className="spinner" /> : null}
          {starting ? 'Starting' : rerun ? 'Update archive' : 'Import messages'}
        </button>
        <button type="button" className="btn quiet lg" disabled={starting} onClick={onBack}>
          {wayBack}
        </button>
      </div>
    </>
  )
}
