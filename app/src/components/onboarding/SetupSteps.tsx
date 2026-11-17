import { recallBridge } from '../../lib/bridge'
import { formatNumber, plural } from '../../lib/format'
import type { SetupImportResult } from '../../types'
import { OtherSource } from './OtherSource'

// The steps of the guide, apart from the import itself. Each one says what is
// happening and offers the way on; OnboardingWizard decides which is showing.

type SourceProps = {
  pickBusy: boolean
  pickError: string
  onPick: () => void
  onManualPath: (path: string) => void
}

export function WelcomeStep({ onStart, onSkip }: { onStart: () => void; onSkip: () => void }) {
  return (
    <>
      <h1>Welcome to Recall</h1>
      <p className="setup-lead">
        Recall reads your iMessage history into a private archive on this Mac. You can search it, ask questions about
        it, and look back through it.
      </p>
      <div className="setup-actions">
        <button type="button" className="btn primary lg" onClick={onStart}>
          Get started
        </button>
        <button type="button" className="btn quiet lg" onClick={onSkip}>
          Not now
        </button>
      </div>
    </>
  )
}

type PermissionStepProps = SourceProps & {
  // true once the reader has been to System Settings and come back
  returnedFromSettings: boolean
  onOpenSettings: () => void
  onRelaunch: () => void
  onSkip: () => void
}

export function PermissionStep({
  returnedFromSettings,
  pickBusy,
  pickError,
  onOpenSettings,
  onRelaunch,
  onPick,
  onManualPath,
  onSkip,
}: PermissionStepProps) {
  return (
    <>
      <h1>Allow Recall to read Messages</h1>
      <p className="setup-lead">
        macOS keeps the Messages database closed to apps until you allow it. Recall only reads it and never changes
        it.
      </p>
      <ol className="setup-list" role="list">
        <li>Open System Settings, then Privacy &amp; Security, then Full Disk Access.</li>
        <li>Turn on Recall. If it is not in the list, add it with the + button.</li>
        <li>Come back to this window.</li>
      </ol>
      <div className="setup-actions">
        {recallBridge.available ? (
          <button type="button" className="btn primary lg" onClick={onOpenSettings}>
            Open System Settings
          </button>
        ) : null}
        <button type="button" className="btn quiet lg" onClick={onSkip}>
          Skip for now
        </button>
      </div>
      {returnedFromSettings ? (
        // macOS applies the permission when an app starts, so the last step is a fresh start
        <div className="notice setup-notice">
          <p>
            <strong>Reopen Recall to finish.</strong> macOS applies Full Disk Access when an app starts. Setup carries
            on from here.
          </p>
          {recallBridge.available ? (
            <button type="button" className="btn" onClick={onRelaunch}>
              Reopen Recall
            </button>
          ) : (
            <p>Quit Recall with Command-Q and open it again.</p>
          )}
        </div>
      ) : (
        <p className="setup-waiting" role="status">
          <span className="spinner" />
          Waiting for access
        </p>
      )}
      <OtherSource busy={pickBusy} error={pickError} onPick={onPick} onPath={onManualPath} />
    </>
  )
}

type NoMessagesStepProps = SourceProps & {
  // missing: no database on this Mac; empty: a database with nothing in it
  variant: 'missing' | 'empty'
  // true after "Check again" found the same thing
  stillNothing: boolean
  onRecheck: () => void
  onSkip: () => void
}

export function NoMessagesStep({
  variant,
  stillNothing,
  pickBusy,
  pickError,
  onRecheck,
  onPick,
  onManualPath,
  onSkip,
}: NoMessagesStepProps) {
  return (
    <>
      <h1>{variant === 'missing' ? 'No Messages database on this Mac' : 'Your Messages history is empty'}</h1>
      <p className="setup-lead">
        {variant === 'missing'
          ? 'Recall looked in ~/Library/Messages and found no chat.db. If your messages are on another Mac, copy chat.db from it and choose the copy below.'
          : 'The database is there with nothing in it. If you signed in to iCloud a moment ago, Messages may still be syncing.'}
      </p>
      {stillNothing ? (
        <p className="setup-note" role="status">
          Still nothing. Try again in a minute.
        </p>
      ) : null}
      <div className="setup-actions">
        <button type="button" className="btn primary lg" onClick={onRecheck}>
          Check again
        </button>
        <button type="button" className="btn quiet lg" onClick={onSkip}>
          Skip for now
        </button>
      </div>
      <OtherSource busy={pickBusy} error={pickError} onPick={onPick} onPath={onManualPath} />
    </>
  )
}

type DbProblemStepProps = SourceProps & {
  // locked: Messages is writing to it; invalid: the file is not a Messages database
  variant: 'invalid' | 'locked'
  // what the check said, in its own words
  detail: string
  onRetry: () => void
  onBack: () => void
}

export function DbProblemStep({
  variant,
  detail,
  pickBusy,
  pickError,
  onRetry,
  onPick,
  onManualPath,
  onBack,
}: DbProblemStepProps) {
  return (
    <>
      <h1>{variant === 'locked' ? 'The Messages database is busy' : 'That file is not a Messages database'}</h1>
      <p className="setup-lead">
        {variant === 'locked'
          ? 'Messages is writing to it. Quit Messages or wait a moment, then try again.'
          : 'Choose the file named chat.db, not chat.db-wal or chat.db-shm. When you copy it, bring all three files so the newest messages come too.'}
      </p>
      {detail ? <p className="setup-note">{detail}</p> : null}
      <div className="setup-actions">
        <button type="button" className="btn primary lg" onClick={onRetry}>
          Try again
        </button>
        <button type="button" className="btn quiet lg" onClick={onBack}>
          Back
        </button>
      </div>
      {variant === 'invalid' ? (
        <OtherSource busy={pickBusy} error={pickError} onPick={onPick} onPath={onManualPath} />
      ) : null}
    </>
  )
}

type DoneStepProps = {
  // what the import brought in; null when the archive was already there
  result: SetupImportResult | null
  rerun: boolean
  onFinish: () => void
}

export function DoneStep({ result, rerun, onFinish }: DoneStepProps) {
  const years =
    result?.firstYear && result?.lastYear
      ? result.firstYear === result.lastYear
        ? String(result.firstYear)
        : `${result.firstYear} to ${result.lastYear}`
      : ''
  const counted = [
    result?.messages ? plural(result.messages, 'message') : '',
    result?.conversations ? plural(result.conversations, 'conversation') : '',
    years,
  ].filter(Boolean)
  return (
    <>
      <h1>Your archive is ready</h1>
      {counted.length ? <p className="setup-lead">{counted.join(', ')}</p> : null}
      {result?.recovered ? (
        <p className="setup-note">{formatNumber(result.recovered)} older messages were recovered from the saved copy.</p>
      ) : null}
      {result?.contactsSkipped ? (
        <p className="setup-note">
          Contacts could not be read, so some people show as numbers. Read them again from Settings at any time.
        </p>
      ) : null}
      <div className="setup-actions">
        <button type="button" className="btn primary lg" onClick={onFinish}>
          {rerun ? 'Done' : 'Show my first words'}
        </button>
      </div>
    </>
  )
}
