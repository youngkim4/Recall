import type { CSSProperties } from 'react'
import { workTime, type Work, type WorkSummary } from '../../lib/chatWork'
import { CheckIcon } from '../Icons'
import { Disclosure } from '../ui/Disclosure'
import { Lamp } from '../ui/Lamp'

// The steps that are done, each with a check: quiet, under one another.
function DoneSteps({ steps }: { steps: string[] }) {
  return (
    <ol className="work-steps" role="list">
      {steps.map((step, index) => (
        <li key={`${index} ${step}`} className="work-step">
          <CheckIcon />
          <span>{step}</span>
        </li>
      ))}
    </ol>
  )
}

// While the server works on a question, after ChatGPT and Copilot: the step
// it is on, with a light passing through the words, and Recall's lamp lit
// beside it in the place of Claude's spark. The steps before it stand above,
// quiet, each with a check. While a conversation is read part by part, a
// thin bar fills under the step and the flame grows with it.
export function Working({ work }: { work: Work | undefined }) {
  const steps = work?.steps ?? []
  const current = steps.at(-1)
  const parts = current?.of ? { read: current.read ?? 0, of: current.of } : null
  return (
    <div className="working is-new" role="status">
      {steps.length > 1 ? <DoneSteps steps={steps.slice(0, -1).map((step) => step.text)} /> : null}
      <div className="work-now">
        <Lamp className="lamp work-lamp" level={parts ? 0.3 + (0.7 * parts.read) / parts.of : 0.45} max={false} lit />
        {/* a new step takes the place of the last with a short fade */}
        <span key={current?.text ?? ''} className="work-text shimmer swap">
          {current?.text ?? 'Thinking'}
        </span>
      </div>
      {parts ? (
        <div className="work-parts" aria-label={`${parts.read} of ${parts.of} parts read`}>
          <span className="work-bar" style={{ '--done': parts.read / parts.of } as CSSProperties}>
            <i />
          </span>
          <span className="work-count">
            {parts.read} of {parts.of}
          </span>
        </div>
      ) : null}
    </div>
  )
}

// Over an answer, once it has started, after ChatGPT's "Thought for 42s":
// how long the work took, which opens to the steps.
export function WorkDone({ work }: { work: WorkSummary }) {
  return (
    <Disclosure className="work-done" summary={`Worked for ${workTime(work.seconds)}`}>
      <DoneSteps steps={work.steps} />
    </Disclosure>
  )
}
