import { useState, type ReactNode } from 'react'
import type { Job, SetupImportResult, SetupStatus } from '../../types'
import { recallBridge } from '../../lib/bridge'

// Presentational steps for the first-run wizard. All state lives in
// OnboardingWizard; these render copy and fire callbacks.

export function OnboardingFrame({
  step,
  children,
}: {
  step: number
  children: ReactNode
}) {
  const labels = ['Welcome', 'Access', 'Import', 'Done']
  return (
    <div className="onboard">
      <div className="onboard-wordmark">Recall</div>
      <div className="onboard-stage">
        <div className="onboard-dots" aria-label={`Step ${step + 1} of ${labels.length}`}>
          {labels.map((label, index) => (
            <span
              key={label}
              className={`onboard-dot${index === step ? ' is-active' : ''}${index < step ? ' is-done' : ''}`}
              title={label}
            />
          ))}
        </div>
        {children}
      </div>
      <p className="onboard-footnote">
        Local-first: your messages are read and stored in files on this Mac. No account, no
        Recall servers. AI answers are optional — they send only the excerpts needed for an
        answer to OpenAI, and only when you ask.
      </p>
    </div>
  )
}

export function WelcomeStep({
  rerun,
  onStart,
  onSkip,
}: {
  rerun: boolean
  onStart: () => void
  onSkip: () => void
}) {
  return (
    <div className="onboard-step">
      <h1 className="onboard-title">{rerun ? 'Update your archive' : 'Welcome to Recall'}</h1>
      <p className="onboard-sub">Every text you’ve kept, ready to revisit.</p>
      <p className="onboard-body">
        {rerun
          ? 'Run the setup again to refresh your archive from Messages — new conversations, new names, everything since last time.'
          : 'Recall reads your iMessage history and turns it into a private archive you can search, ask questions of, and wander through — first words, old promises, the group chat’s greatest hits.'}
      </p>
      <div className="onboard-aside">
        Everything happens here. Your archive is built and stored on this Mac, in files you can
        see. Nothing about you or your messages is sent anywhere to make this work.
      </div>
      <div className="onboard-actions">
        <button type="button" className="button primary" onClick={onStart}>
          Get started
        </button>
        <button type="button" className="button ghost" onClick={onSkip}>
          Not now
        </button>
      </div>
    </div>
  )
}

export function FallbackSourceCard({
  onPick,
  onManualPath,
  busy,
  error,
}: {
  onPick: () => void
  onManualPath: (path: string) => void
  busy: boolean
  error: string
}) {
  const [manual, setManual] = useState('')
  return (
    <div className="onboard-fallback">
      <div className="onboard-fallback-title">Rather not grant access?</div>
      <p className="onboard-fallback-body">
        That works too. Point Recall at a copy of your Messages database instead — one you copied
        yourself, or brought over from another Mac.
      </p>
      {recallBridge.available ? (
        <button type="button" className="button ghost" onClick={onPick} disabled={busy}>
          {busy ? 'Checking that file…' : 'Choose a file…'}
        </button>
      ) : (
        <div className="onboard-manual-path">
          <input
            type="text"
            value={manual}
            placeholder="/path/to/a/copy/of/chat.db"
            onChange={(event) => setManual(event.target.value)}
          />
          <button
            type="button"
            className="button ghost"
            disabled={busy || !manual.trim()}
            onClick={() => onManualPath(manual.trim())}
          >
            {busy ? 'Checking…' : 'Use this file'}
          </button>
        </div>
      )}
      {error ? <div className="onboard-error">{error}</div> : null}
      <p className="onboard-hint">
        Your database lives at ~/Library/Messages/chat.db. In Finder use Go, then Go to Folder,
        copy chat.db somewhere handy, and choose the copy here.
      </p>
    </div>
  )
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
}: {
  returnedFromSettings: boolean
  pickBusy: boolean
  pickError: string
  onOpenSettings: () => void
  onRelaunch: () => void
  onPick: () => void
  onManualPath: (path: string) => void
  onSkip: () => void
}) {
  return (
    <div className="onboard-step">
      <h1 className="onboard-title">Allow Recall to read your Messages</h1>
      <p className="onboard-body">
        macOS keeps your Messages database off-limits to apps — which is exactly right. To build
        your archive, Recall needs Full Disk Access, a permission you grant once in System
        Settings. Recall only ever reads the Messages database. It never changes it, and nothing
        is uploaded.
      </p>
      <ol className="onboard-checklist">
        <li>Open System Settings → Privacy &amp; Security → Full Disk Access.</li>
        <li>Find Recall in the list — use the + button if it isn’t there — and switch it on.</li>
        <li>Come back here. Recall notices on its own.</li>
      </ol>
      <div className="onboard-actions">
        {recallBridge.available ? (
          <button type="button" className="button primary" onClick={onOpenSettings}>
            Open System Settings
          </button>
        ) : (
          <code className="onboard-path-line">
            System Settings → Privacy &amp; Security → Full Disk Access
          </code>
        )}
        <button type="button" className="button ghost" onClick={onSkip}>
          Skip for now
        </button>
      </div>
      <div className="onboard-waiting">
        <span className="onboard-pulse" aria-hidden="true" />
        Watching for access…
      </div>
      {returnedFromSettings ? (
        <div className="onboard-relaunch">
          <div className="onboard-fallback-title">One more step: reopen Recall</div>
          <p className="onboard-fallback-body">
            macOS applies Full Disk Access when an app starts fresh. Quit and reopen Recall —
            setup will pick up right where you left off.
          </p>
          {recallBridge.available ? (
            <button type="button" className="button" onClick={onRelaunch}>
              Relaunch Recall
            </button>
          ) : (
            <p className="onboard-hint">Quit Recall (Cmd+Q) and open it again — setup resumes on its own.</p>
          )}
        </div>
      ) : null}
      <FallbackSourceCard onPick={onPick} onManualPath={onManualPath} busy={pickBusy} error={pickError} />
    </div>
  )
}

export function NoMessagesStep({
  variant,
  pickBusy,
  pickError,
  onRecheck,
  onPick,
  onManualPath,
  onSkip,
  stillNothing,
}: {
  variant: 'missing' | 'empty'
  pickBusy: boolean
  pickError: string
  onRecheck: () => void
  onPick: () => void
  onManualPath: (path: string) => void
  onSkip: () => void
  stillNothing: boolean
}) {
  return (
    <div className="onboard-step">
      <h1 className="onboard-title">
        {variant === 'missing' ? 'We couldn’t find a Messages database' : 'Your Messages history looks empty'}
      </h1>
      <p className="onboard-body">
        {variant === 'missing'
          ? 'This Mac doesn’t seem to have an iMessage history yet — it usually lives at ~/Library/Messages/chat.db. If your messages are on another Mac, copy chat.db over and choose it below.'
          : 'The database is there, but there’s nothing in it yet. If you just signed in, iCloud may still be syncing — give it a little time and check again.'}
      </p>
      {stillNothing ? <p className="onboard-hint">Still nothing yet — it’s worth another look in a minute.</p> : null}
      <div className="onboard-actions">
        <button type="button" className="button" onClick={onRecheck}>
          Check again
        </button>
        <button type="button" className="button ghost" onClick={onSkip}>
          Skip for now
        </button>
      </div>
      <FallbackSourceCard onPick={onPick} onManualPath={onManualPath} busy={pickBusy} error={pickError} />
    </div>
  )
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
}: {
  variant: 'invalid' | 'locked'
  detail: string
  pickBusy: boolean
  pickError: string
  onRetry: () => void
  onPick: () => void
  onManualPath: (path: string) => void
  onBack: () => void
}) {
  return (
    <div className="onboard-step">
      <h1 className="onboard-title">
        {variant === 'locked' ? 'The Messages database is busy' : 'That file isn’t a Messages database'}
      </h1>
      <p className="onboard-body">
        {variant === 'locked'
          ? 'Messages is writing to it right now. Quit Messages, or just wait a moment, and try again.'
          : 'Choose the file named chat.db itself — not chat.db-wal or chat.db-shm. If you’re copying it, bring all three files along so the newest messages come too.'}
      </p>
      {detail ? <p className="onboard-hint">{detail}</p> : null}
      <div className="onboard-actions">
        <button type="button" className="button primary" onClick={onRetry}>
          Try again
        </button>
        <button type="button" className="button ghost" onClick={onBack}>
          Back
        </button>
      </div>
      {variant === 'invalid' ? (
        <FallbackSourceCard onPick={onPick} onManualPath={onManualPath} busy={pickBusy} error={pickError} />
      ) : null}
    </div>
  )
}

function approxCount(value: number | null | undefined): string {
  if (!value || value <= 0) return ''
  if (value >= 1000) return `${Math.round(value / 1000).toLocaleString()},000`
  return value.toLocaleString()
}

export function ImportStep({
  status,
  job,
  rerun,
  onStart,
  onRetry,
  onBack,
}: {
  status: SetupStatus
  job: Job | null
  rerun: boolean
  onStart: () => void
  onRetry: () => void
  onBack: () => void
}) {
  const running = job && (job.status === 'queued' || job.status === 'running')
  const failed = job && job.status === 'failed'
  const db = status.db
  const statLine = db.approxMessages
    ? `About ${approxCount(db.approxMessages)} messages${
        db.conversations ? ` across ${db.conversations.toLocaleString()} conversations` : ''
      }${db.firstYear ? `, going back to ${db.firstYear}` : ''}.`
    : ''

  if (failed) {
    const interrupted = (job?.error || '').includes('interrupted')
    return (
      <div className="onboard-step">
        <h1 className="onboard-title">That didn’t work</h1>
        {job?.error ? <p className="onboard-body onboard-error-text">{job.error}</p> : null}
        <p className="onboard-body">
          {interrupted ? '' : 'Nothing was changed — your Messages stayed exactly as they were.'}
        </p>
        <div className="onboard-actions">
          <button type="button" className="button primary" onClick={onRetry}>
            Try again
          </button>
          <button type="button" className="button ghost" onClick={onBack}>
            Back
          </button>
        </div>
      </div>
    )
  }

  if (running) {
    const step = job?.progress?.step || 1
    const phases = [
      'Reading your Messages database',
      'Matching names from Contacts',
      'Getting your archive ready',
    ]
    const lastLog = job?.logs?.length ? job.logs[job.logs.length - 1] : null
    const logText = typeof lastLog === 'string' ? lastLog : lastLog?.message || ''
    return (
      <div className="onboard-step">
        <h1 className="onboard-title">Building your archive</h1>
        <ol className="onboard-phases">
          {phases.map((label, index) => {
            const n = index + 1
            const stateClass = n < step ? 'is-done' : n === step ? 'is-active' : ''
            return (
              <li key={label} className={`onboard-phase ${stateClass}`}>
                <span className="onboard-phase-num">{n}</span>
                {label}
              </li>
            )
          })}
        </ol>
        <p className="onboard-hint">This usually takes under a minute. Big archives can take a few.</p>
        {logText ? <code className="onboard-log-line">{logText}</code> : null}
      </div>
    )
  }

  return (
    <div className="onboard-step">
      <h1 className="onboard-title">{rerun ? 'Refresh your archive' : 'Build your archive'}</h1>
      {statLine ? <p className="onboard-stat">{statLine}</p> : null}
      <p className="onboard-body">
        Recall will copy your message history into its own archive — a file that lives with the
        app on this Mac. Your Messages app and its database aren’t touched. Nothing is uploaded.
      </p>
      <div className="onboard-actions">
        <button type="button" className="button primary" onClick={onStart}>
          {rerun ? 'Refresh my archive' : 'Import my messages'}
        </button>
        <button type="button" className="button ghost" onClick={onBack}>
          Back
        </button>
      </div>
    </div>
  )
}

export function DoneStep({
  result,
  rerun,
  onFinish,
}: {
  result: SetupImportResult | null
  rerun: boolean
  onFinish: () => void
}) {
  const statParts = []
  if (result?.messages) statParts.push(`${result.messages.toLocaleString()} messages`)
  if (result?.conversations) statParts.push(`${result.conversations.toLocaleString()} conversations`)
  if (result?.firstYear && result?.lastYear) statParts.push(`${result.firstYear}–${result.lastYear}`)
  return (
    <div className="onboard-step">
      <h1 className="onboard-title">Your archive is ready.</h1>
      {statParts.length ? <p className="onboard-stat">{statParts.join(' · ')}</p> : null}
      {result?.recovered ? (
        <p className="onboard-hint">
          Recovered {result.recovered.toLocaleString()} older messages from your saved snapshot.
        </p>
      ) : null}
      {result?.contactsSkipped ? (
        <p className="onboard-hint">
          Contacts was skipped, so some people show as numbers. Fix it anytime: Settings, then
          Refresh contacts.
        </p>
      ) : null}
      <div className="onboard-actions">
        <button type="button" className="button primary" onClick={onFinish}>
          {rerun ? 'Back to Recall' : 'Show me my first words'}
        </button>
      </div>
    </div>
  )
}
