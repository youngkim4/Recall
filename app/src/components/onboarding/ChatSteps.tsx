import { formatMoney } from '../../lib/format'
import { STEP_LABELS, costCeiling, progressLine, stepProblems, type ChatPhase } from '../../lib/firstRun'
import { useFirstRun } from '../../lib/useFirstRun'
import { useKey } from '../../lib/useKey'
import type { FirstRunStatus, Job } from '../../types'
import { KeyForm } from '../firstrun/KeyForm'
import { CheckIcon } from '../Icons'
import { ImportProgress } from './ImportProgress'

// The two steps of the guide after the import: the API key, then getting
// chat ready. Both can be left for later; search works without either.

type KeyStepProps = {
  busy: boolean
  error: string
  onSave: (typed: string) => Promise<boolean>
  onLater: () => void
}

export function KeyStep({ busy, error, onSave, onLater }: KeyStepProps) {
  return (
    <>
      <h1>Add your API key</h1>
      <p className="setup-lead">
        Chat and reports need it. It is saved in a file on this Mac and used only when you ask. Search works without it.
      </p>
      {/* the guide puts the focus on the step's heading, as on every step */}
      <KeyForm busy={busy} error={error} onSave={onSave} large />
      <div className="setup-actions">
        <button type="button" className="btn quiet lg" onClick={onLater}>
          Later
        </button>
      </div>
    </>
  )
}

type ChatStepProps = {
  phase: ChatPhase
  status: FirstRunStatus | null
  job: Job | null
  starting: boolean
  refusal: string
  onStart: () => void
  // leaves the guide; a job on its way goes on without it
  onFinish: () => void
  // the words of the button that leaves the guide once chat is ready
  finishLabel: string
}

// What chat needs, what of it is ready, and what the rest costs at most. One
// button builds it all; the reader can leave while it runs.
export function ChatStep({ phase, status, job, starting, refusal, onStart, onFinish, finishLabel }: ChatStepProps) {
  if (phase === 'ready') {
    return (
      <>
        <h1>Chat is ready</h1>
        <p className="setup-lead">Recall keeps it up to date by itself after each import.</p>
        <div className="setup-actions">
          <button type="button" className="btn primary lg" onClick={onFinish}>
            {finishLabel}
          </button>
        </div>
      </>
    )
  }

  if (phase === 'running' && job) {
    const step = job.progress?.step ?? 1
    return (
      <>
        <h1>Getting chat ready</h1>
        <p className="setup-lead">This can take a while on a large archive. You can use Recall in the meantime.</p>
        <ImportProgress phases={Object.values(STEP_LABELS)} current={step} said={progressLine(job)} />
        <div className="setup-actions">
          <button type="button" className="btn lg" onClick={onFinish}>
            Continue in the background
          </button>
        </div>
      </>
    )
  }

  const cost = status ? costCeiling(status.estimatedCost) : 0
  const problems = status ? stepProblems(status) : []
  return (
    <>
      <h1>Get chat ready</h1>
      <p className="setup-lead">
        Recall reads your archive once, so chat can search by meaning, remember each conversation and know the people in
        it.
      </p>
      {status ? <ChatParts status={status} /> : null}
      {phase === 'not_ready' || phase === 'failed' ? (
        // a usage note, not a bill: the user may charge for usage one day
        <p className="setup-size">Up to {formatMoney(cost)} of usage, once</p>
      ) : null}
      {problems.map((problem) => (
        <p key={problem} className="setup-note">
          {problem}
        </p>
      ))}
      {status && !status.key.present ? (
        <p className="setup-note">Chat needs an API key first. Add it in Settings &gt; Archive &gt; API key.</p>
      ) : null}
      {phase === 'no_messages' ? <p className="setup-note">The archive holds no messages yet.</p> : null}
      {phase === 'unknown' ? <p className="setup-note">What chat needs could not be checked.</p> : null}
      {phase === 'failed' && job?.error ? (
        <p className="notice bad" role="alert">
          {job.error}
        </p>
      ) : null}
      {refusal ? (
        <p className="notice bad" role="alert">
          {refusal}
        </p>
      ) : null}
      <div className="setup-actions">
        {phase === 'not_ready' || phase === 'failed' ? (
          <button
            type="button"
            className="btn primary lg"
            disabled={starting || !status?.key.present}
            onClick={onStart}
          >
            {phase === 'failed' ? 'Try again' : 'Get chat ready'}
          </button>
        ) : null}
        <button type="button" className="btn quiet lg" onClick={onFinish}>
          Later
        </button>
      </div>
    </>
  )
}

// The key step with its own state: it asks where the key is only once it shows.
export function KeyStage({ onDone }: { onDone: () => void }) {
  const { busy, error, save } = useKey()
  return (
    <KeyStep
      busy={busy}
      error={error}
      onSave={async (typed) => {
        const saved = await save(typed)
        if (saved) onDone()
        return saved
      }}
      onLater={onDone}
    />
  )
}

// The chat step with its own state: it asks what chat needs only once it shows.
export function ChatStage({
  messagesPath,
  onFinish,
  finishLabel,
}: {
  messagesPath: string
  onFinish: () => void
  finishLabel: string
}) {
  const { phase, status, job, starting, refusal, start } = useFirstRun(messagesPath)
  return (
    <ChatStep
      phase={phase}
      status={status}
      job={job}
      starting={starting}
      refusal={refusal}
      onStart={() => void start()}
      onFinish={onFinish}
      finishLabel={finishLabel}
    />
  )
}

// the three things chat needs, a check on each one that is ready
function ChatParts({ status }: { status: FirstRunStatus }) {
  return (
    <ul className="setup-phases" role="list" aria-label="What chat needs">
      {status.steps.map((step) => {
        const ready = step.state === 'ready'
        return (
          <li key={step.name} data-state={ready ? 'done' : undefined}>
            <span className="setup-phase-mark" aria-hidden="true">
              {ready ? <CheckIcon /> : null}
            </span>
            {STEP_LABELS[step.name]}
            {ready ? <span className="sr-only">, ready</span> : null}
          </li>
        )
      })}
    </ul>
  )
}
