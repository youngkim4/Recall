import { useEffect, useRef, type ReactNode } from 'react'

const STEPS = ['Welcome', 'Access', 'Import', 'Done']

type SetupFrameProps = {
  // which of the four steps is showing, from 0; null for an update, which is not a sequence
  step: number | null
  // what the step is showing now; a change of it arrives
  phase: string
  children: ReactNode
}

// The guide that builds the archive takes the whole window: one step at a
// time in a narrow column, how far along it is above, and what happens to the
// messages below, where it can be read at every step.
export function SetupFrame({ step, phase, children }: SetupFrameProps) {
  const shown = useRef<HTMLDivElement>(null)
  const first = useRef(true)

  // The button that led here went away with the step it was on. The new step
  // takes the focus, and a screen reader starts from its heading.
  useEffect(() => {
    if (first.current) {
      first.current = false
      return
    }
    shown.current?.focus()
  }, [phase])

  return (
    <main className="setup">
      <div className="setup-column">
        {step === null ? null : (
          <ol className="setup-steps" role="list" aria-label={`Step ${step + 1} of ${STEPS.length}: ${STEPS[step]}`}>
            {STEPS.map((name, index) => (
              <li key={name} data-state={index === step ? 'now' : index < step ? 'done' : undefined} />
            ))}
          </ol>
        )}
        {/* what a step shows arrives with a short rise; the key starts it again */}
        <div key={phase} ref={shown} className="setup-step arrive" tabIndex={-1}>
          {children}
        </div>
      </div>
      <p className="setup-privacy">
        Your messages stay in files on this Mac. There is no account. AI answers are optional and send only the
        excerpts an answer needs, and only when you ask.
      </p>
    </main>
  )
}
